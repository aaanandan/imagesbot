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
async function uploadToDrive(filePath, newName) {
  const auth = new google.auth.GoogleAuth({
    keyFile: "credentials.json",
    scopes: ["https://www.googleapis.com/auth/drive.file"],
  });
  const drive = google.drive({ version: "v3", auth });

  const res = await drive.files.create({
    requestBody: {
      name: newName,
      parents: [process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID],
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
  const chatId = msg.chat.id;
  const fileId = msg.photo[msg.photo.length - 1].file_id;
  const file = await bot.getFile(fileId);

  const url = `https://api.telegram.org/file/bot${process.env.TELEGRAM_BOT_TOKEN}/${file.file_path}`;
  const folder = path.join("downloads", String(chatId));
  await fs.ensureDir(folder);

  const localPath = path.join(folder, `${Date.now()}.jpg`);
  const res = await fetch(url);
  const buffer = Buffer.from(await res.arrayBuffer());
  await fs.writeFile(localPath, buffer);

  bot.sendMessage(chatId, "📥 Image received, processing...");

  try {
    const hasPerson = await detectPerson(localPath);
    const ocrText = await extractLargestText(localPath);

    const now = new Date();
    const newName = formatFileName(now, now.getMinutes(), ocrText, hasPerson);

    const renamedPath = path.join(folder, newName);
    await fs.rename(localPath, renamedPath);

    const driveLink = await uploadToDrive(renamedPath, newName);
    bot.sendMessage(chatId, `✅ Processed: *${newName}*\n🔗 [View on Drive](${driveLink})`, { parse_mode: "Markdown" });

  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "❌ Error processing image");
  }
});
