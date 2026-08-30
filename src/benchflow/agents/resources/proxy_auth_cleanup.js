"use strict";

const crypto = require("crypto");
const fs = require("fs");

const credentialTargets = JSON.parse(process.argv[1]);
const settingsTargets = JSON.parse(process.argv[2]);
const constants = fs.constants;
const directoryFlags =
  constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW;

function openDirectoryNoFollow(parts) {
  let descriptor = fs.openSync("/", directoryFlags);
  try {
    for (const part of parts) {
      const next = fs.openSync(
        `/proc/self/fd/${descriptor}/${part}`,
        directoryFlags,
      );
      fs.closeSync(descriptor);
      descriptor = next;
    }
    return descriptor;
  } catch (error) {
    fs.closeSync(descriptor);
    throw error;
  }
}

function splitTarget(target) {
  if (typeof target !== "string" || !target.startsWith("/")) {
    throw new Error("cleanup target must be an absolute path");
  }
  const parts = target.split("/").filter(Boolean);
  const basename = parts.pop();
  if (!basename || parts.includes("..")) {
    throw new Error(`invalid cleanup target: ${target}`);
  }
  return {parts, basename};
}

function openParent(target) {
  const {parts, basename} = splitTarget(target);
  try {
    return {descriptor: openDirectoryNoFollow(parts), basename};
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

function removeCredentialNoFollow(target) {
  const parent = openParent(target);
  if (parent === null) return;
  const {descriptor, basename} = parent;
  try {
    const descriptorPath = `/proc/self/fd/${descriptor}/${basename}`;
    try {
      fs.unlinkSync(descriptorPath);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    try {
      fs.lstatSync(descriptorPath);
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }
    throw new Error(`credential remained after deletion: ${target}`);
  } finally {
    fs.closeSync(descriptor);
  }
}

function parseSettings(raw, target) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`invalid JSON settings file ${target}: ${error.message}`);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`settings file must contain a JSON object: ${target}`);
  }
  return parsed;
}

function sanitizeSettingsNoFollow(spec) {
  if (
    spec === null ||
    typeof spec !== "object" ||
    typeof spec.path !== "string" ||
    !Array.isArray(spec.drop_keys) ||
    spec.drop_keys.some((key) => typeof key !== "string" || !key)
  ) {
    throw new Error("invalid settings sanitization specification");
  }

  const parent = openParent(spec.path);
  if (parent === null) return;
  const {descriptor, basename} = parent;
  const descriptorPath = `/proc/self/fd/${descriptor}/${basename}`;
  let sourceDescriptor;
  let tempPath;
  try {
    try {
      sourceDescriptor = fs.openSync(
        descriptorPath,
        constants.O_RDONLY | constants.O_NOFOLLOW,
      );
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }

    const sourceStat = fs.fstatSync(sourceDescriptor);
    if (!sourceStat.isFile()) {
      throw new Error(`settings target is not a regular file: ${spec.path}`);
    }
    const settings = parseSettings(
      fs.readFileSync(sourceDescriptor, "utf8"),
      spec.path,
    );
    fs.closeSync(sourceDescriptor);
    sourceDescriptor = undefined;

    let changed = false;
    for (const key of spec.drop_keys) {
      if (Object.prototype.hasOwnProperty.call(settings, key)) {
        delete settings[key];
        changed = true;
      }
    }
    if (!changed) return;

    const tempName = `.benchflow-sanitized-${process.pid}-${crypto
      .randomBytes(8)
      .toString("hex")}`;
    tempPath = `/proc/self/fd/${descriptor}/${tempName}`;
    const tempDescriptor = fs.openSync(
      tempPath,
      constants.O_WRONLY |
        constants.O_CREAT |
        constants.O_EXCL |
        constants.O_NOFOLLOW,
      0o600,
    );
    try {
      fs.fchownSync(tempDescriptor, sourceStat.uid, sourceStat.gid);
      fs.fchmodSync(tempDescriptor, sourceStat.mode & 0o777);
      fs.writeFileSync(tempDescriptor, `${JSON.stringify(settings, null, 2)}\n`);
      fs.fsyncSync(tempDescriptor);
    } finally {
      fs.closeSync(tempDescriptor);
    }
    fs.renameSync(tempPath, descriptorPath);
    tempPath = undefined;

    const verifiedDescriptor = fs.openSync(
      descriptorPath,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
    try {
      const verified = parseSettings(
        fs.readFileSync(verifiedDescriptor, "utf8"),
        spec.path,
      );
      for (const key of spec.drop_keys) {
        if (Object.prototype.hasOwnProperty.call(verified, key)) {
          throw new Error(`credential setting remained after sanitization: ${key}`);
        }
      }
    } finally {
      fs.closeSync(verifiedDescriptor);
    }
  } finally {
    if (sourceDescriptor !== undefined) fs.closeSync(sourceDescriptor);
    if (tempPath !== undefined) {
      try {
        fs.unlinkSync(tempPath);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    fs.closeSync(descriptor);
  }
}

if (!Array.isArray(credentialTargets) || !Array.isArray(settingsTargets)) {
  throw new Error("cleanup arguments must be arrays");
}
for (const target of credentialTargets) removeCredentialNoFollow(target);
for (const spec of settingsTargets) sanitizeSettingsNoFollow(spec);
