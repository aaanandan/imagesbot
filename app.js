import TelegramBot from "node-telegram-bot-api";
import fs from "fs-extra";
import path from "path";
import dotenv from "dotenv";
import { google } from "googleapis";
import * as tf from "@tensorflow/tfjs-node";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { createCanvas, loadImage } from "canvas";
import Tesseract from "tesseract.js";

dotenv.config();

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

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
  return predictions.some(p => p.class === "person");
}

// --- OCR largest text block ---
async function extractLargestText(imagePath) {
  const { data: { text } } = await Tesseract.recognize(imagePath, "eng");
  const blocks = text.split("\n").map(t => t.trim()).filter(t => t.length > 0);
  return blocks.sort((a, b) => b.length - a.length)[0] || "unknown";
}

// --- Rename file ---
function formatFileName(date, minute, ocrText, hasPerson) {
  // example: 5sep-08.10am-person
  const d = date.toLocaleDateString("en-GB", { day: "numeric", month: "short" }).replace(" ", "").toLowerCase();
  const t = date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: true }).toLowerCase().replace(" ", "");
  return `${d}-${t}${hasPerson ? "-person" : ""}.jpg`;
}

// --- Upload to Google Drive ---
async function uploadToDrive(filePath, newName, uploadFolder) {
  const auth = new google.auth.GoogleAuth({
    keyFile: "credentials.json",
    scopes: ["https://www.googleapis.com/auth/drive.file"],
  });
  const drive = google.drive({ version: "v3", auth });

  const res = await drive.files.create({
    requestBody: {
      name: newName,
      parents: uploadFolder,
    },
    media: {
      mimeType: "image/jpeg",
      body: fs.createReadStream(filePath),
    },
  });

  const fileId = res.data.id;
  await drive.permissions.create({ fileId, requestBody: { role: "reader", type: "anyone" } });
  return `https://drive.google.com/file/d/${fileId}/view`;
}

// --- Telegram Bot Handler ---
bot.on("photo", async (msg) => {
  
  const { chatId, batchFolder:folder, localPaths } = await processImageBatch(bot, msg);
  const uploadFolder = createFolder(`${Date.now()}`,process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID);

  localPaths.forEach(async (localPath) => {
    try {
      const hasPerson = await detectPerson(localPath);
      const ocrText = await extractLargestText(localPath);
  
      const parsedDate = parseDateTimeFromText(ocrText) || new Date(); // fallback to system time
      const newName = formatFileName(parsedDate, parsedDate.getMinutes(), ocrText, hasPerson);

      const renamedPath = path.join(folder, newName);
      await fs.rename(newName, renamedPath);
  
      const driveLink = await uploadToDrive(renamedPath, newName);
      
    } catch (err) {
      console.error(err);
      bot.sendMessage(chatId, "❌ Error processing image");
    }

  })
  
  const driveLink = `https://drive.google.com/drive/folders/${uploadFolder}`;
  bot.sendMessage(chatId, `✅ Processed: *${localPaths}*\n🔗 [View on Drive](${driveLink})`, { parse_mode: "Markdown" });
  
});


function parseDateTimeFromText(text) {
  // Normalize text
  const clean = text.toLowerCase().replace(/\s+/g, " ");

  // Regex for patterns like "5 sep 8:10am" or "05/09 08:10"
  const dateRegex = /(\d{1,2})(?:st|nd|rd|th)?[\/\-\s]?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?[a-z]*[\/\-\s]?(\d{2,4})?/;
  const timeRegex = /(\d{1,2})(?::(\d{2}))?\s?(am|pm)?/;

  const dateMatch = clean.match(dateRegex);
  const timeMatch = clean.match(timeRegex);

  let day = 1, month = 0, year = new Date().getFullYear(); // defaults
  let hour = 0, minute = 0;

  if (dateMatch) {
    day = parseInt(dateMatch[1], 10);

    if (dateMatch[2]) {
      const months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec"];
      month = months.indexOf(dateMatch[2].slice(0,3));
      if (month < 0) month = 0;
    }
    if (dateMatch[3]) {
      year = parseInt(dateMatch[3], 10);
      if (year < 100) year += 2000; // handle "24" → "2024"
    }
  }

  if (timeMatch) {
    hour = parseInt(timeMatch[1], 10);
    minute = timeMatch[2] ? parseInt(timeMatch[2], 10) : 0;
    if (timeMatch[3] === "pm" && hour < 12) hour += 12;
    if (timeMatch[3] === "am" && hour === 12) hour = 0;
  }

  return new Date(year, month, day, hour, minute);
}

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


async function processImageBatch(bot, msg) {
  const chatId = msg.chat.id;

  // Each "photo" array contains multiple sizes of the same image.
  // We'll take the highest resolution version for each photo.
  const photos = msg.photo || [];

  // Create a batch folder for this chat
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