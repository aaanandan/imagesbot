import fs from "fs-extra";
import path from "path";
import fetch from "node-fetch";
import TelegramBot from "node-telegram-bot-api";
import { google } from "googleapis";
import * as exifParser from "exif-parser";
import tesseract from "tesseract.js";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs-node";
import "dotenv/config";

const __dirname = path.resolve();

// ---- Config ----
const token = process.env.TELEGRAM_BOT_TOKEN;
const uploadFolder = process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID;
const bot = new TelegramBot(token, { polling: true });
let model; // coco-ssd model cache
const albumStore = {}; // temporary store for albums

// ---- Helpers ----
function pad2(n) {
  return String(n).padStart(2, "0");
}

async function loadModel() {
  if (!model) {
    model = await cocoSsd.load();
    console.log("✅ COCO-SSD model loaded");
  }
  return model;
}

async function detectPerson(imagePath) {
  try {
    const mdl = await loadModel();
    const img = fs.readFileSync(imagePath);
    const decoded = tf.node.decodeImage(img, 3);
    const preds = await mdl.detect(decoded);
    decoded.dispose();
    return preds.some((p) => p.class === "person" && p.score > 0.5);
  } catch (err) {
    console.error("⚠️ Person detection failed", err);
    return false;
  }
}

async function getPhotoDate(localPath) {
  try {
    const buffer = await fs.readFile(localPath);
    const parser = exifParser.create(buffer);
    const exif = parser.parse();
    if (exif.tags && exif.tags.DateTimeOriginal) {
      return new Date(exif.tags.DateTimeOriginal * 1000);
    }
  } catch (err) {
    console.error("No EXIF date", err);
  }
  return null;
}

async function extractLargestText(imagePath) {
  try {
    const { data: { text } } = await tesseract.recognize(imagePath, "eng");
    return text.trim().split(/\n+/).sort((a, b) => b.length - a.length)[0] || "";
  } catch (err) {
    console.error("OCR failed", err);
    return "";
  }
}

function parseDateTimeFromText(text) {
  const now = new Date();
  // fallback values
  return {
    day: pad2(now.getDate()),
    month: pad2(now.getMonth() + 1),
    year: now.getFullYear(),
    hour: pad2(now.getHours()),
    minute: pad2(now.getMinutes())
  };
}

async function uploadToDrive(filePath, newName, folderId) {
  const auth = new google.auth.GoogleAuth({
    keyFile: process.env.GDRIVE_KEY_FILE,
    scopes: ["https://www.googleapis.com/auth/drive.file"],
  });
  const drive = google.drive({ version: "v3", auth });

  const res = await drive.files.create({
    requestBody: {
      name: newName,
      parents: [folderId],
    },
    media: {
      mimeType: "image/jpeg",
      body: fs.createReadStream(filePath),
    },
    fields: "id",
  });
  return `https://drive.google.com/file/d/${res.data.id}/view`;
}

// ---- Core Processing ----
async function processImageBatch(msg, photos) {
  const batchFolder = `temp_${msg.message_id}`;
  await fs.mkdir(batchFolder, { recursive: true });
  console.log("📂 Created temp folder:", batchFolder);

  const tasks = photos.map(async (photo, i) => {
    const fileId = photo.file_id;
    const fileLink = await bot.getFileLink(fileId);
    const localPath = path.join(batchFolder, `photo_${i}.jpg`);
    const res = await fetch(fileLink);
    const buffer = Buffer.from(await res.arrayBuffer());
    await fs.writeFile(localPath, buffer);

    let newName;
    let renamedPath = localPath;

    const photoDate = await getPhotoDate(localPath);
    if (photoDate) {
      newName = `${pad2(photoDate.getHours())}.${pad2(photoDate.getMinutes())} ${pad2(photoDate.getDate())}${pad2(photoDate.getMonth() + 1)}${photoDate.getFullYear()}.jpg`;
    } else {
      const ocrText = await extractLargestText(localPath);
      const { day, month, year, hour, minute } = parseDateTimeFromText(ocrText);
      const safeOcr = ocrText.replace(/[^a-zA-Z0-9]/g, "").slice(0, 20);
      newName = `${hour}.${minute} ${day}${month}${year}_${safeOcr}.jpg`;
    }

    renamedPath = path.join(batchFolder, newName);
    await fs.rename(localPath, renamedPath);

    const hasPerson = await detectPerson(renamedPath);

    const driveLink = await uploadToDrive(renamedPath, newName, uploadFolder);
    await bot.sendMessage(
      msg.chat.id,
      `✅ Processed: *${newName}*\n👤 Person detected: ${hasPerson ? "Yes" : "No"}\n🔗 [View on Drive](${driveLink})`,
      { parse_mode: "Markdown" }
    );
  });

  await Promise.allSettled(tasks);

  const driveLink = `https://drive.google.com/drive/folders/${uploadFolder}`;
  await bot.sendMessage(msg.chat.id, `📂 All images uploaded: ${driveLink}`);
  await fs.remove(batchFolder);
  console.log("🧹 Temp folder cleaned:", batchFolder);
}

// ---- Telegram Listener ----
bot.on("message", async (msg) => {
  try {
    console.log("📩 Message received..");

    if (msg.photo) {
      const bestPhoto = msg.photo[msg.photo.length - 1];
      if (msg.media_group_id) {
        if (!albumStore[msg.media_group_id]) {
          albumStore[msg.media_group_id] = [];
        }
        albumStore[msg.media_group_id].push(bestPhoto);

        setTimeout(async () => {
          const batch = albumStore[msg.media_group_id];
          if (batch && batch.length > 0) {
            console.log(`📸 Processing album ${msg.media_group_id} (${batch.length} photos)`);
            await processImageBatch(msg, batch);
            delete albumStore[msg.media_group_id];
          }
        }, 5000);
      } else {
        await processImageBatch(msg, [bestPhoto]);
      }
    } else if (msg.document && msg.document.mime_type.startsWith("image/")) {
      await processImageBatch(msg, [msg.document]);
    }
  } catch (err) {
    console.error("❌ Error processing message:", err);
    await bot.sendMessage(msg.chat.id, "❌ Failed to process image(s).");
  }
});

console.log("🤖 Bot started...");
