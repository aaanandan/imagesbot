import TelegramBot from "node-telegram-bot-api";
import fs from "fs-extra";
import path from "path";
import dotenv from "dotenv";
import { google } from "googleapis";
import * as tf from "@tensorflow/tfjs-node";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { createCanvas, loadImage } from "canvas";
import Tesseract from "tesseract.js";
import fetch from "node-fetch";
import ExifParser from "exif-parser";
import http from "http";

dotenv.config();

// --- Telegram bot setup ---
const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

// --- Google Drive setup ---
const auth = new google.auth.GoogleAuth({
  keyFile: "credentials.json", // ✅ service account JSON
  scopes: ["https://www.googleapis.com/auth/drive.file"]
});
const drive = google.drive({ version: "v3", auth });

// --- Load ML model once ---
let cocoModel;
async function loadModel() {
  if (!cocoModel) {
    cocoModel = await cocoSsd.load();
    console.log("✅ COCO-SSD model loaded");
  }
  return cocoModel;
}

// --- Detect person in image ---
async function detectPerson(imagePath) {
  const mdl = await loadModel();
  const img = await loadImage(imagePath);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const tensor = tf.browser.fromPixels(canvas);
  const predictions = await mdl.detect(tensor);
  return predictions.some((p) => p.class === "person");
}

// --- OCR largest text block ---
async function extractLargestText(imagePath) {
  const {
    data: { text }
  } = await Tesseract.recognize(imagePath, "eng");
  const blocks = text
    .split("\n")
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  return blocks.sort((a, b) => b.length - a.length)[0] || "unknown";
}

// --- Extract EXIF date ---
function getPhotoDate(filePath) {
  const buffer = fs.readFileSync(filePath);
  const parser = ExifParser.create(buffer);
  const result = parser.parse();
  return result.tags.DateTimeOriginal
    ? new Date(result.tags.DateTimeOriginal * 1000)
    : null;
}

// --- Upload to Google Drive ---
async function uploadToDrive(filePath, newName, uploadFolder) {
  const res = await drive.files.create({
    requestBody: {
      name: newName,
      parents: [uploadFolder]
    },
    media: {
      mimeType: "image/jpeg",
      body: fs.createReadStream(filePath)
    }
  });

  const fileId = res.data.id;
  await drive.permissions.create({
    fileId,
    requestBody: { role: "reader", type: "anyone" }
  });
  return `https://drive.google.com/file/d/${fileId}/view`;
}

// --- Create Google Drive folder ---
async function createFolder(name, parentId = null) {
  const fileMetadata = {
    name,
    mimeType: "application/vnd.google-apps.folder"
  };
  if (parentId) fileMetadata.parents = [parentId];

  const folder = await drive.files.create({
    resource: fileMetadata,
    fields: "id"
  });
  return folder.data.id;
}

// --- Album cache ---
const albumStore = {};

// --- Handle incoming messages ---
bot.on("message", async (msg) => {
  try {
    console.log("📩 Message received..");

    if (msg.photo) {
      console.log("📸 Message has photos..");
      const bestPhoto = msg.photo[msg.photo.length - 1];

      if (msg.media_group_id) {
        if (!albumStore[msg.media_group_id]) {
          albumStore[msg.media_group_id] = [];
        }
        albumStore[msg.media_group_id].push(bestPhoto);

        setTimeout(async () => {
          const batch = albumStore[msg.media_group_id];
          if (batch && batch.length > 0) {
            console.log(
              `📸 Processing album ${msg.media_group_id} (${batch.length} photos)`
            );
            await processImageBatch(msg, batch);
            delete albumStore[msg.media_group_id];
          }
        }, 5000); // wait to collect all photos
      } else {
        await processImageBatch(msg, [bestPhoto]);
      }
    } else if (msg.document && msg.document.mime_type.startsWith("image/")) {
      console.log("📑 Message has image document..");
      await processImageBatch(msg, [msg.document]);
    }
  } catch (err) {
    console.error("❌ Error processing message:", err);
    await bot.sendMessage(msg.chat.id, "❌ Failed to process image(s).");
  }
});

// --- Process one batch (album or single photo) ---
async function processImageBatch(msg, photos) {
  const batchFolder = path.join(
    "downloads",
    String(msg.chat.id),
    String(Date.now())
  );
  await fs.ensureDir(batchFolder);
  console.log(`📂 Created temp folder: ${batchFolder}`);

  // --- Create Drive upload folder with date/time + photo count ---
  const now = new Date();
  const folderName = `${now.toISOString().replace(/[:.]/g, "-")}_${photos.length}photos`;
  const uploadFolder = await createFolder(
    folderName,
    process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID
  );

  const tasks = photos.map(async (photo, i) => {
    try {
      const file = await bot.getFile(photo.file_id);
      const url = `https://api.telegram.org/file/bot${process.env.TELEGRAM_BOT_TOKEN}/${file.file_path}`;
      const localPath = path.join(batchFolder, `image_${i + 1}.jpg`);

      // download
      const res = await fetch(url);
      const buffer = Buffer.from(await res.arrayBuffer());
      await fs.writeFile(localPath, buffer);

      // detect person
      // const hasPerson = await detectPerson(localPath);

      // date or OCR
      const photoDate = getPhotoDate(localPath);
      let newName;
      if (photoDate) {
        const d = String(photoDate.getDate()).padStart(2, "0");
        const m = String(photoDate.getMonth() + 1).padStart(2, "0");
        const y = photoDate.getFullYear();
        const hh = String(photoDate.getHours()).padStart(2, "0");
        const mm = String(photoDate.getMinutes()).padStart(2, "0");
        // newName = `${d}${m}${y}_${hh}${mm}${hasPerson ? "-person" : ""}.jpg`;
        newName = `${d}${m}${y}_${hh}${mm}.jpg`;
      } else {
        const ocrText = await extractLargestText(localPath);
        newName = `${Date.now()}_${ocrText.replace(/\s+/g, "_")}.jpg`;
        // newName = `${Date.now()}_${ocrText.replace(/\s+/g, "_")}${
        //   hasPerson ? "-person" : ""
        // }.jpg`;
      }

      const renamedPath = path.join(batchFolder, newName);
      await fs.rename(localPath, renamedPath);

      const driveLink = await uploadToDrive(renamedPath, newName, uploadFolder);

      await bot.sendMessage(
        msg.chat.id,
        `✅ Processed: *${newName}*\n🔗 [View on Drive](${driveLink})`,
        { parse_mode: "Markdown" }
      );
    } catch (err) {
      console.error("❌ Error processing photo:", err);
      await bot.sendMessage(msg.chat.id, "❌ Error processing photo");
    }
  });

  await Promise.allSettled(tasks);

  // final album link
  const driveFolderLink = `https://drive.google.com/drive/folders/${uploadFolder}`;
  await bot.sendMessage(
    msg.chat.id,
    `📂 All images uploaded: ${driveFolderLink}`
  );

  await deleteFolder(batchFolder);
}

// --- Delete local folder ---
async function deleteFolder(folderPath) {
  try {
    await fs.remove(folderPath);
    console.log(`🧹 Deleted folder: ${folderPath}`);
  } catch (err) {
    console.error(`❌ Error deleting folder ${folderPath}:`, err);
  }
}

// --- Tiny HTTP server (for Render/Heroku health checks) ---
const PORT = process.env.PORT || 3000;
http
  .createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("✅ Bot is running\n");
  })
  .listen(PORT, () =>
    console.log(`🚀 Server running at http://localhost:${PORT}/`)
  );
