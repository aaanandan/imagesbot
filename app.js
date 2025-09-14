import TelegramBot from "node-telegram-bot-api";
import fs from "fs-extra";
import path from "path";
import dotenv from "dotenv";
import { google } from "googleapis";
import * as tf from "@tensorflow/tfjs-node";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { createCanvas, loadImage } from "canvas";
import Tesseract from "tesseract.js";
import fetch from "node-fetch"; // ✅ required if Node < 18
import ExifParser from "exif-parser";
import http from "http";
import { text } from "stream/consumers";

dotenv.config();

// --- Telegram bot setup ---
const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

// --- Google Drive setup ---
const auth = new google.auth.GoogleAuth({
  keyFile: "credentials.json",
  scopes: ["https://www.googleapis.com/auth/drive.file"],
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
    data: { text },
  } = await Tesseract.recognize(imagePath, "eng");
  const blocks = text
    .split("\n")
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  return blocks.sort((a, b) => b.length - a.length)[0] || "unknown";
}

function getPhotoDate(filePath) {
  const buffer = fs.readFileSync(filePath);
  const parser = ExifParser.create(buffer);
  const result = parser.parse();
  return  result.tags.DateTimeOriginal
  ? new Date(result.tags.DateTimeOriginal * 1000) // convert seconds → ms
  : null;

}

// --- Rename file ---
function formatFileName(date, minute, ocrText, hasPerson) {
  const d = date
    .toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
    })
    .replace(" ", "")
    .toLowerCase();
  const t = date
    .toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    })
    .toLowerCase()
    .replace(" ", "");
  return `${d}-${t}${hasPerson ? "-person" : ""}.jpg`;
}

// --- Upload to Google Drive ---
async function uploadToDrive(filePath, newName, uploadFolder) {
  const res = await drive.files.create({
    requestBody: {
      name: newName,
      parents: [uploadFolder], // ✅ must be array
    },
    media: {
      mimeType: "image/jpeg",
      body: fs.createReadStream(filePath),
    },
  });

  const fileId = res.data.id;
  await drive.permissions.create({
    fileId,
    requestBody: { role: "reader", type: "anyone" },
  });
  return `https://drive.google.com/file/d/${fileId}/view`;
}

// --- Telegram Bot Handler ---
bot.on("photo", async (msg) => {
  try {
    const { chatId, batchFolder: folder, localPaths } = await processImageBatch(
      bot,
      msg
    );

    // create Drive folder
    const uploadFolder = await createFolder(
      `${Date.now()}`,
      process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID
    );

    for (const localPath of localPaths) {
      try {
        const photoDate = await getPhotoDate(localPath);    
        let renamedPath = localPath;
        if(photoDate){
          const newName = `${photoDate.getHours()}.${photoDate.getMinutes()} ${photoDate.getDay()}${photoDate.getMonth()}${photoDate.getFullYear()}`;
          renamedPath = path.join(folder, newName);
          await fs.rename(localPath, renamedPath); // ✅ fixed rename
        }else{
          const ocrText = extractLargestText(renamedPath);
          const { day, month, year, hour, minute} = parseDateTimeFromText(ocrText);
          const newName = `${hour}.${minute} ${day}${month}${year}`;
          renamedPath = path.join(folder, newName);
          await fs.rename(localPath, renamedPath); // ✅ fixed rename
        }

        const driveLink = await uploadToDrive(
          renamedPath,
          newName,
          uploadFolder
        );
        await bot.sendMessage(
          chatId,
          `✅ Processed: *${newName}*\n🔗 [View on Drive](${driveLink})`,
          { parse_mode: "Markdown" }
        );
      } catch (err) {
        console.error(err);
        await bot.sendMessage(chatId, "❌ Error processing image");
      }
    }

    const driveLink = `https://drive.google.com/drive/folders/${uploadFolder}`;
    await bot.sendMessage(chatId, `📂 All images uploaded: ${driveLink}`);

    // cleanup local
    await deleteFolder(folder);
  } catch (err) {
    console.error("Batch error:", err);
  }
});

// --- Date parsing from OCR ---
function parseDateTimeFromText(text) { 
  if (!text) return null;

  // normalize
  let clean = String(text).toLowerCase().replace(/\r?\n/g, " ").replace(/\s+/g, " ").trim();

  // OCR fixes *between digits*
  clean = clean.replace(/(\d)[oO](\d)/g, "$10$2").replace(/(\d)[lI](\d)/g, "$11$2");

  const monthMap = {
    jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
    jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11
  };

  // 1) textual date + time
  const textualRe = /(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s*)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[,\s]+(\d{2,4}))?(?:[,\s\-@]*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?/i;
  const m1 = clean.match(textualRe);
  if (m1) {
    const day = parseInt(m1[1], 10);
    const monKey = m1[2].slice(0, 3).toLowerCase();
    const month = monthMap[monKey] ?? null;   // only from text
    const year = m1[3] ? parseInt(m1[3], 10) : null;
    let hour = m1[4] ? parseInt(m1[4], 10) : 0;
    const minute = m1[5] ? parseInt(m1[5], 10) : 0;
    const ampm = (m1[6] || "").toLowerCase();
    if (ampm === "pm" && hour < 12) hour += 12;
    if (ampm === "am" && hour === 12) hour = 0;
    return { day, month, year, hour, minute };
  }

  // 2) numeric date dd/mm[/yyyy] or dd-mm-yyyy or dd.mm.yyyy
  const numericDateRe = /(\d{1,2})[\/\-.](\d{1,2})(?:[\/\-.](\d{2,4}))?/;
  const m2 = clean.match(numericDateRe);
  if (m2) {
    const day = parseInt(m2[1], 10);
    const month = parseInt(m2[2], 10) - 1; // numeric month from text
    const year = m2[3] ? parseInt(m2[3], 10) : null;
    return { day, month, year, hour: 0, minute: 0 };
  }

  // 3) time-only
  const timeOnlyRe = /(\d{1,2}):(\d{2})\s*(am|pm)?/i;
  const timeOnlyAlt = /(\d{1,2})\s*(am|pm)/i;
  const m3 = clean.match(timeOnlyRe) || clean.match(timeOnlyAlt);
  if (m3) {
    let hour = parseInt(m3[1], 10);
    const minute = m3[2] ? parseInt(m3[2], 10) : 0;
    const ampm = (m3[3] || "").toLowerCase();
    if (ampm === "pm" && hour < 12) hour += 12;
    if (ampm === "am" && hour === 12) hour = 0;
    return { day: null, month: null, year: null, hour, minute };
  }

  return null;
}


// --- Create Google Drive folder ---
async function createFolder(name, parentId = null) {
  const fileMetadata = {
    name,
    mimeType: "application/vnd.google-apps.folder",
  };
  if (parentId) fileMetadata.parents = [parentId];

  const folder = await drive.files.create({
    resource: fileMetadata,
    fields: "id",
  });
  return folder.data.id;
}

// --- Process incoming Telegram images ---
async function processImageBatch(bot, msg) {
  const chatId = msg.chat.id;
  const photos = msg.photo || [];

  const batchFolder = path.join("downloads", String(chatId), String(Date.now()));
  await fs.ensureDir(batchFolder);

  const localPaths = [];

  for (let i = 0; i < photos.length; i++) {
    const fileId = photos[i].file_id;
    const file = await bot.getFile(fileId);

    const url = `https://api.telegram.org/file/bot${process.env.TELEGRAM_BOT_TOKEN}/${file.file_path}`;
    const localPath = path.join(batchFolder, `image_${i + 1}.jpg`);

    const res = await fetch(url);
    const buffer = Buffer.from(await res.arrayBuffer());
    await fs.writeFile(localPath, buffer);

    localPaths.push(localPath);
  }

  console.log(`Downloaded ${localPaths.length} images for chat ${chatId}`);
  return { chatId, batchFolder, localPaths };
}

// --- Delete local folder ---
async function deleteFolder(folderPath) {
  try {
    await fs.remove(folderPath);
    console.log(`Deleted folder: ${folderPath}`);
  } catch (err) {
    console.error(`Error deleting folder ${folderPath}:`, err);
  }
}


const PORT = process.env.PORT || 3000;

const server = http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("Hello! Your Node.js HTTP server is running.\n");
});

server.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}/`);
});