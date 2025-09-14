import TelegramBot from "node-telegram-bot-api";
import fs from "fs";
import path from "path";
import { google } from "googleapis";
import fetch from "node-fetch";
import ExifParser from "exif-parser";
import Tesseract from "tesseract.js";
import * as Human from "@vladmandic/human";
import http from "http";

// ===== CONFIG =====
const TELEGRAM_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const DRIVE_FOLDER_ID =  process.env.GOOGLE_DRIVE_PARENT_FOLDER_ID;
const CREDENTIALS_PATH = "credentials.json"; // service account credentials

// ===== INIT =====
const bot = new TelegramBot(TELEGRAM_TOKEN, { polling: true });

const auth = new google.auth.GoogleAuth({
  keyFile: CREDENTIALS_PATH,
  scopes: ["https://www.googleapis.com/auth/drive.file"],
});
const drive = google.drive({ version: "v3", auth });

// Human config (lightweight person detection)
const human = new Human.Human({
  modelBasePath: "https://vladmandic.github.io/human/models/",
  face: { enabled: true },
  body: { enabled: true },
});

// ===== HELPERS =====
const pad2 = (n) => String(n).padStart(2, "0");

function parseDateTimeFromText(text) {
  if (!text) return null;

  const clean = String(text).toLowerCase();
  const dateRegex = /(\d{1,2})[\/\-\. ](\d{1,2})(?:[\/\-\. ](\d{2,4}))?/;
  const timeRegex = /(\d{1,2}):(\d{2})(?::(\d{2}))?/;

  const dateMatch = clean.match(dateRegex);
  const timeMatch = clean.match(timeRegex);

  if (!dateMatch) return null;

  let day = parseInt(dateMatch[1]);
  let month = parseInt(dateMatch[2]) - 1;
  let year = dateMatch[3] ? parseInt(dateMatch[3]) : new Date().getFullYear();

  let hour = timeMatch ? parseInt(timeMatch[1]) : 0;
  let minute = timeMatch ? parseInt(timeMatch[2]) : 0;

  if (year < 100) year += 2000;

  return new Date(year, month, day, hour, minute);
}

async function extractExifDate(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const parser = ExifParser.create(buffer);
    const result = parser.parse();
    return result.tags.DateTimeOriginal
      ? new Date(result.tags.DateTimeOriginal * 1000)
      : null;
  } catch {
    return null;
  }
}

async function extractOcrText(filePath) {
  try {
    const { data } = await Tesseract.recognize(filePath, "eng");
    return data.text.trim().replace(/\s+/g, " ").slice(0, 30); // keep suffix short
  } catch {
    return null;
  }
}

async function detectPerson(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const result = await human.detect(buffer);

    const hasPerson =
      (result.face && result.face.length > 0) ||
      (result.body && result.body.length > 0);

    return hasPerson;
  } catch {
    return false;
  }
}

async function uploadToDrive(filePath, fileName) {
  const fileMetadata = {
    name: fileName,
    parents: [DRIVE_FOLDER_ID],
  };
  const media = {
    mimeType: "image/jpeg",
    body: fs.createReadStream(filePath),
  };

  const file = await drive.files.create({
    resource: fileMetadata,
    media,
    fields: "id, webViewLink",
  });

  return file.data.webViewLink;
}

// ===== CORE =====
async function processImages(localPaths, messageText) {
  return await Promise.allSettled(
    localPaths.map(async (localPath) => {
      let finalDate = null;

      // 1. Try EXIF
      finalDate = await extractExifDate(localPath);

      // 2. Try OCR if no EXIF
      let ocrText = "";
      if (!finalDate) {
        ocrText = await extractOcrText(localPath);
        finalDate = parseDateTimeFromText(ocrText);
      }

      // 3. Try message text if still missing
      if (!finalDate) {
        finalDate = parseDateTimeFromText(messageText);
      }

      // 4. Default now
      if (!finalDate) {
        finalDate = new Date();
      }

      const year = finalDate.getFullYear();
      const month = finalDate.getMonth();
      const day = finalDate.getDate();
      const hour = finalDate.getHours();
      const minute = finalDate.getMinutes();

      // Person detection
      const hasPerson = await detectPerson(localPath);

      // Always force JPG
      const newName = `${pad2(hour)}.${pad2(minute)}_${pad2(day)}${pad2(
        month + 1
      )}${year}${
        ocrText ? "_" + ocrText.replace(/[^a-zA-Z0-9]/g, "") : ""
      }${!hasPerson ? "_NoPerson" : ""}.jpg`;

      const tempJpg = localPath + ".jpg";

      // Ensure it's saved as JPG for upload
      fs.copyFileSync(localPath, tempJpg);

      const link = await uploadToDrive(tempJpg, newName);

      // Cleanup
      fs.unlinkSync(localPath);
      fs.unlinkSync(tempJpg);

      return { file: newName, link, hasPerson };
    })
  );
}

async function processImageBatch(msg, photos) {
  const batchFolder = `temp_${msg.message_id}`;
  fs.mkdirSync(batchFolder, { recursive: true });

  try {
    const downloads = photos.map(async (photo, i) => {
      const fileId = photo.file_id;
      const file = await bot.getFile(fileId);
      const fileUrl = `https://api.telegram.org/file/bot${TELEGRAM_TOKEN}/${file.file_path}`;

      const res = await fetch(fileUrl);
      const buffer = await res.buffer();

      const ext = path.extname(file.file_path) || ".jpg";
      const localPath = path.join(batchFolder, `image_${i + 1}${ext}`);
      fs.writeFileSync(localPath, buffer);
      return localPath;
    });

    const localPaths = await Promise.all(downloads);
    const results = await processImages(localPaths, msg.caption || msg.text);

    const success = results
      .filter((r) => r.status === "fulfilled")
      .map((r) => r.value);

    if (success.length) {
      const links = success
        .map(
          (s) =>
            `${s.hasPerson ? "✅ Person" : "⚠️ No Person"} — *${s.file}*\n🔗 ${
              s.link
            }`
        )
        .join("\n\n");

      await bot.sendMessage(msg.chat.id, `📂 Processed:\n\n${links}`, {
        parse_mode: "Markdown",
      });
    } else {
      await bot.sendMessage(msg.chat.id, "❌ No images processed.");
    }
  } finally {
    fs.rmSync(batchFolder, { recursive: true, force: true });
  }
}

// ===== TELEGRAM HANDLER =====
bot.on("message", async (msg) => {
  if (msg.photo) {
    await processImageBatch(msg, msg.photo);
  }
});


const PORT = process.env.PORT || 3000; 
const server = http.createServer((req, res) => { res.writeHead(200, { "Content-Type": "text/plain" }); 
res.end("Hello! Your Node.js HTTP server is running.\n"); }); 
server.listen(PORT, () => { console.log(`🚀 Server running at http:// localhost:${PORT}`); });
