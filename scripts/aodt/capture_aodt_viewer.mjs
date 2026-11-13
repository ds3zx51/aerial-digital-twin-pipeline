#!/usr/bin/env node
/**
 * Capture a native AODT Viewer RU-filtered raypath figure in headless Chromium.
 *
 * Required environment:
 *   MINIO_ACCESS_KEY
 *   MINIO_SECRET_KEY
 *
 * Optional environment:
 *   AODT_VIEWER_URL, AODT_VIEWER_DB, AODT_VIEWER_YAML, AODT_VIEWER_OUTPUT
 *   AODT_VIEWER_RU_ID, AODT_CATALOG_URI, AODT_S3_ENDPOINT, AODT_S3_BUCKET
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const viewerUrl = process.env.AODT_VIEWER_URL || "http://aodt-viewer:5173";
const database =
  process.env.AODT_VIEWER_DB ||
  "tokyo_fixed_8ru_50ue_viewer_20260717T144706Z";
const ymlPath =
  process.env.AODT_VIEWER_YAML || "/work/sim_config_submitted.yml";
const outputDir = process.env.AODT_VIEWER_OUTPUT || "/work/output";
const ruId = Number(process.env.AODT_VIEWER_RU_ID || "8");
const catalogUri =
  process.env.AODT_CATALOG_URI || "http://nessie:19120/iceberg";
const s3Endpoint = process.env.AODT_S3_ENDPOINT || "http://minio:9000";
const s3BucketName = process.env.AODT_S3_BUCKET || "aerial-data";
const accessKey = process.env.MINIO_ACCESS_KEY || "";
const secretKey = process.env.MINIO_SECRET_KEY || "";

if (!accessKey || !secretKey) {
  throw new Error("MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required.");
}
if (!Number.isInteger(ruId) || ruId < 1) {
  throw new Error(`Invalid AODT_VIEWER_RU_ID: ${ruId}`);
}
if (!fs.existsSync(ymlPath)) {
  throw new Error(`AODT Viewer YAML does not exist: ${ymlPath}`);
}

fs.mkdirSync(outputDir, { recursive: true });
const ymlContent = fs.readFileSync(ymlPath, "utf8");
const fullScreenshot = path.join(
  outputDir,
  `aodt_ru${ruId - 1}_native_viewer_full.png`,
);
const sceneScreenshot = path.join(
  outputDir,
  `aodt_ru${ruId - 1}_native_viewer_scene.png`,
);
const statusPath = path.join(outputDir, "capture_status.json");
const browserLogPath = path.join(outputDir, "browser_console.log");
const browserLines = [];

const minioSettings = {
  catalogType: "rest",
  glueRegion: "us-east-1",
  catalogUri,
  s3Endpoint,
  s3BucketName,
  s3Provider: "minio",
  accessKey,
  secretKey,
};
const minioConfig = {
  baseUrl: "",
  database,
  catalogUri: `${catalogUri.replace(/\/+$/, "")}/`,
  catalogType: "rest",
  s3BucketName,
  s3Endpoint,
  accessKey,
  secretKey,
};
const filterState = {
  enabledRuIds: [ruId],
  enabledUeIds: [],
  allRuEnabled: false,
  allUeEnabled: true,
};
const cameraState = {
  position: {
    longitude: 139.7332,
    latitude: 35.6532,
    height: 1450,
  },
  orientation: {
    heading: 0.72,
    pitch: -0.58,
    roll: 0,
  },
};

let browser;
let page;
try {
  browser = await chromium.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu-sandbox",
      "--enable-webgl",
      "--ignore-gpu-blocklist",
      "--enable-unsafe-swiftshader",
      "--use-gl=angle",
      "--use-angle=swiftshader",
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    colorScheme: "dark",
  });
  page = await context.newPage();
  page.setDefaultTimeout(120_000);
  page.setDefaultNavigationTimeout(120_000);
  page.on("console", (message) => {
    const line = `[console:${message.type()}] ${message.text()}`;
    browserLines.push(line);
    console.log(line);
  });
  page.on("pageerror", (error) => {
    const line = `[pageerror] ${error.stack || error.message}`;
    browserLines.push(line);
    console.error(line);
  });
  page.on("requestfailed", (request) => {
    const line = `[requestfailed] ${request.url()} :: ${
      request.failure()?.errorText || "unknown"
    }`;
    browserLines.push(line);
    console.error(line);
  });

  await page.addInitScript(
    ({
      minioSettingsValue,
      minioConfigValue,
      databaseValue,
      filterStateValue,
      cameraStateValue,
      ymlContentValue,
    }) => {
      localStorage.clear();
      localStorage.setItem(
        "minio_settings",
        JSON.stringify(minioSettingsValue),
      );
      localStorage.setItem("minio_config", JSON.stringify(minioConfigValue));
      localStorage.setItem("yml-editor-content", ymlContentValue);
      localStorage.setItem(
        "yml-editor-filename",
        "sim_config_submitted.yml",
      );
      localStorage.setItem("active_right_tab", "Settings");
      localStorage.setItem(
        `raypath_filters_${databaseValue}`,
        JSON.stringify(filterStateValue),
      );
      localStorage.setItem(
        "layer_visibility",
        JSON.stringify({ rayPathsVisible: true, tilesetsVisible: true }),
      );
      localStorage.setItem(
        "scenario_ray_visualization",
        JSON.stringify({
          raysSparsity: 1,
          maxVisibleRayPaths: 500,
          maxDynamicRangeDB: 100,
        }),
      );
      localStorage.setItem(
        "cesium_camera_state",
        JSON.stringify(cameraStateValue),
      );
    },
    {
      minioSettingsValue: minioSettings,
      minioConfigValue: minioConfig,
      databaseValue: database,
      filterStateValue: filterState,
      cameraStateValue: cameraState,
      ymlContentValue: ymlContent,
    },
  );

  console.log(`Opening ${viewerUrl}`);
  await page.goto(viewerUrl, { waitUntil: "domcontentloaded" });
  await page.locator("#cesiumContainer canvas").waitFor({ timeout: 120_000 });

  const settingsTab = page.getByRole("button", {
    name: "Settings",
    exact: true,
  });
  await settingsTab.click();

  const databaseOption = page.locator(`option[value="${database}"]`);
  // Native <option> nodes remain "hidden" until their select is expanded.
  // Presence in the DOM is sufficient for selectOption().
  await databaseOption.waitFor({ state: "attached", timeout: 120_000 });
  const databaseSelect = page
    .locator("select")
    .filter({ has: databaseOption })
    .first();
  await databaseSelect.selectOption(database);

  const loadButton = page.getByRole("button", { name: "Load", exact: true });
  await loadButton.waitFor({ state: "visible" });
  console.log(`Loading Iceberg namespace ${database}`);
  await loadButton.click();
  await page
    .getByRole("button", { name: "Refresh", exact: true })
    .waitFor({ timeout: 360_000 });

  const raysTab = page.getByRole("button", { name: "Rays", exact: true });
  await raysTab.click();
  const targetRu = page.getByRole("button", {
    name: `RU${ruId}`,
    exact: true,
  });
  await targetRu.waitFor({ timeout: 240_000 });

  const ruButtons = page.locator('button[title^="RU "]');
  const ruCount = await ruButtons.count();
  const availableRuIds = [];
  for (let index = 0; index < ruCount; index += 1) {
    const button = ruButtons.nth(index);
    const title = (await button.getAttribute("title")) || "";
    const match = /^RU (\d+): (.+)$/.exec(title);
    if (!match) continue;
    const currentId = Number(match[1]);
    availableRuIds.push(currentId);
    const isEnabled = match[2].includes("Click to hide");
    if (currentId === ruId && !isEnabled) {
      await button.click();
    } else if (currentId !== ruId && isEnabled) {
      await button.click();
    }
  }

  await page.waitForTimeout(20_000);
  await page.screenshot({ path: fullScreenshot, fullPage: true });

  // Both sidebars expose a collapse control. Use the first currently visible one
  // instead of triggering Playwright strict-mode ambiguity.
  const collapseButton = page
    .locator('button[title="Collapse sidebar"]:visible')
    .first();
  if (await collapseButton.isVisible()) {
    await collapseButton.click();
    await page.waitForTimeout(3_000);
  }
  await page.screenshot({ path: sceneScreenshot, fullPage: true });

  const finalFilter = await page.evaluate((databaseValue) => {
    const raw = localStorage.getItem(`raypath_filters_${databaseValue}`);
    return raw ? JSON.parse(raw) : null;
  }, database);
  const status = {
    schema_version: "aodt-native-viewer-capture-v1",
    status: "pass",
    generated_at_utc: new Date().toISOString(),
    viewer_url: viewerUrl,
    database,
    scenario_ru_id: ruId - 1,
    aodt_ru_id: ruId,
    available_aodt_ru_ids: availableRuIds.sort((a, b) => a - b),
    filter: finalFilter,
    screenshots: [
      {
        path: fullScreenshot,
        bytes: fs.statSync(fullScreenshot).size,
        includes_native_legend_and_ru_filter: true,
      },
      {
        path: sceneScreenshot,
        bytes: fs.statSync(sceneScreenshot).size,
        includes_native_legend_and_ru_filter: false,
      },
    ],
    rendering: {
      browser: "Chromium/Playwright headless",
      webgl: "ANGLE SwiftShader requested",
      note: "A100 computes AODT raypaths; the headless browser may use software WebGL.",
    },
  };
  fs.writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  fs.writeFileSync(browserLogPath, `${browserLines.join("\n")}\n`);
  console.log(`CAPTURE_STATUS=pass`);
  console.log(`FULL_SCREENSHOT=${fullScreenshot}`);
  console.log(`SCENE_SCREENSHOT=${sceneScreenshot}`);
} catch (error) {
  fs.writeFileSync(browserLogPath, `${browserLines.join("\n")}\n`);
  const failure = {
    schema_version: "aodt-native-viewer-capture-v1",
    status: "fail",
    generated_at_utc: new Date().toISOString(),
    viewer_url: viewerUrl,
    database,
    scenario_ru_id: ruId - 1,
    aodt_ru_id: ruId,
    error: error?.stack || String(error),
  };
  fs.writeFileSync(statusPath, `${JSON.stringify(failure, null, 2)}\n`);
  if (page) {
    try {
      await page.screenshot({
        path: path.join(outputDir, "capture_failure.png"),
        fullPage: true,
      });
    } catch {
      // Preserve the original failure.
    }
  }
  console.error(error);
  process.exitCode = 2;
} finally {
  if (browser) await browser.close();
}
