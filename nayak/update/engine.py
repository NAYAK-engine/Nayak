"""
nayak/update/engine.py — NAYAK Update Engine operational backend.

Provides the concrete implementation of the :class:`UpdateBase` for NAYAK's Layer 8.
Handles checking for new skills or core updates, downloading packages, verifying
integrity via SHA256 checksums, extracting archives, and managing the local
skills lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import zipfile
from pathlib import Path
from typing import List

import httpx

from nayak.core.bus import EventType, bus
from nayak.core.registry import ModuleStatus, registry
from nayak.update.base import SkillPackage, UpdateBase, UpdateStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NayakUpdater
# ─────────────────────────────────────────────────────────────────────────────

class NayakUpdater(UpdateBase):
    """The concrete operational update engine for NAYAK.

    Responsible for OTA (Over-The-Air) updates and dynamic skill management.
    Supports real HTTP downloads, SHA256 checksum verification, and zip
    extraction into the local skills directory.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skills_dir: str = "skills"
        self._registry_url: str = "https://raw.githubusercontent.com/nayak-engine/nayak/main/registry.json"
        self._backup_dir: str = ".nayak_backups"

        # Ensure required directories exist
        os.makedirs(self._skills_dir, exist_ok=True)
        os.makedirs(self._backup_dir, exist_ok=True)

    @property
    def name(self) -> str:
        """Registry identifier for this operational engine."""
        return "nayak-updater"

    # ── Integrity Verification ────────────────────────────────────────────────

    def verify_checksum(self, package: SkillPackage, file_path: str) -> bool:
        """Verify the SHA256 checksum of a downloaded file against the package manifest.

        Args:
            package:   The :class:`SkillPackage` whose ``checksum`` field is
                       compared against the file on disk.
            file_path: Absolute or relative path to the downloaded file.

        Returns:
            ``True`` if the checksums match or if no checksum is specified
            (skip check).  ``False`` if the checksums differ.
        """
        if not package.checksum:
            logger.debug(
                "UpdateEngine: No checksum provided for '%s' — skipping verification",
                package.name,
            )
            return True

        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            computed = sha256.hexdigest()

            if computed == package.checksum:
                logger.info(
                    "UpdateEngine: Checksum MATCH for '%s' (%s)",
                    package.name, computed[:16],
                )
                return True
            else:
                logger.error(
                    "UpdateEngine: Checksum MISMATCH for '%s' — "
                    "expected=%s, got=%s",
                    package.name, package.checksum[:16], computed[:16],
                )
                return False

        except Exception as exc:
            logger.error(
                "UpdateEngine: Checksum verification failed for '%s': %s",
                package.name, exc,
            )
            return False

    # ── Core Update Logic ─────────────────────────────────────────────────────

    async def check_updates(self) -> List[SkillPackage]:
        """Query the remote registry for available module or skill updates.

        Returns:
            A list of ``SkillPackage`` objects that are not yet installed or have
            a newer version available.
        """
        logger.info("UpdateEngine: Checking for updates at %s", self._registry_url)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self._registry_url)
                response.raise_for_status()
                packages_data = response.json()

            new_pending = []
            for pkg_dict in packages_data:
                pkg = SkillPackage(
                    name=pkg_dict["name"],
                    version=pkg_dict["version"],
                    description=pkg_dict.get("description", ""),
                    author=pkg_dict.get("author", "community"),
                    size_bytes=pkg_dict.get("size_bytes", 0),
                    checksum=pkg_dict.get("checksum", ""),
                    metadata=pkg_dict.get("metadata", {})
                )

                # Check if already installed with same or newer version
                is_new = True
                for installed in self.installed_packages.values():
                    if installed.name == pkg.name and installed.version >= pkg.version:
                        is_new = False
                        break

                if is_new:
                    new_pending.append(pkg)

            self.pending_updates = new_pending
            logger.info("UpdateEngine: Found %d pending updates", len(new_pending))
            return self.pending_updates

        except Exception as exc:
            logger.warning("UpdateEngine: Failed to check updates: %s", exc)
            return []

    async def download(self, package: SkillPackage) -> bool:
        """Download a specific SkillPackage from its remote URL.

        Fetches the file via HTTP, saves it as ``skills/{package.name}.zip``,
        verifies the download is non-empty, and optionally checks its SHA256
        checksum against the manifest.

        Args:
            package: The package to download.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        package.status = UpdateStatus.DOWNLOADING
        logger.info(
            "UpdateEngine: Starting download for %s v%s",
            package.name, package.version,
        )

        try:
            download_url = package.metadata.get("url")
            if not download_url:
                logger.error(
                    "UpdateEngine: No URL provided for package '%s'",
                    package.name,
                )
                package.status = UpdateStatus.FAILED
                return False

            # Download the file
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(download_url)
                resp.raise_for_status()

            # Write to disk
            file_path = os.path.join(self._skills_dir, f"{package.name}.zip")
            with open(file_path, "wb") as f:
                f.write(resp.content)

            # Verify file exists and is non-empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.error(
                    "UpdateEngine: Downloaded file is empty or missing for '%s'",
                    package.name,
                )
                package.status = UpdateStatus.FAILED
                return False

            package.size_bytes = os.path.getsize(file_path)
            package.install_path = file_path

            # Verify checksum if provided
            if not self.verify_checksum(package, file_path):
                logger.error(
                    "UpdateEngine: Checksum verification failed for '%s' — aborting",
                    package.name,
                )
                os.remove(file_path)
                package.status = UpdateStatus.FAILED
                return False

            logger.info(
                "UpdateEngine: Downloaded '%s' (%d bytes) → %s",
                package.name, package.size_bytes, file_path,
            )

            package.status = UpdateStatus.INSTALLING
            return True

        except Exception as exc:
            logger.error(
                "UpdateEngine: Download failed for %s: %s",
                package.name, exc,
            )
            package.status = UpdateStatus.FAILED
            return False

    async def install(self, package: SkillPackage) -> bool:
        """Extract and register a pre-downloaded package into the active system.

        If a zip file exists at ``skills/{package.name}.zip``, it is extracted
        into ``skills/{package.name}/`` and the archive is deleted.  The package
        is then added to the installed registry.

        Args:
            package: The package to install.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        package.status = UpdateStatus.INSTALLING
        logger.info(
            "UpdateEngine: Installing %s v%s",
            package.name, package.version,
        )

        try:
            zip_path = os.path.join(self._skills_dir, f"{package.name}.zip")
            extract_dir = os.path.join(self._skills_dir, package.name)

            if os.path.exists(zip_path):
                # Extract the zip archive
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                logger.info(
                    "UpdateEngine: Extracted '%s' → %s",
                    package.name, extract_dir,
                )

                # Remove the zip after successful extraction
                os.remove(zip_path)
                logger.debug(
                    "UpdateEngine: Removed archive %s after extraction", zip_path,
                )

            # Register in installed packages
            package.install_path = extract_dir
            self.installed_packages[package.package_id] = package
            package.status = UpdateStatus.COMPLETE

            await self.emit_update_event(package)
            logger.info(
                "UpdateEngine: Successfully installed %s v%s",
                package.name, package.version,
            )
            return True

        except Exception as exc:
            logger.error(
                "UpdateEngine: Installation failed for %s: %s",
                package.name, exc,
            )
            package.status = UpdateStatus.FAILED
            return False

    async def rollback(self, package_id: str) -> bool:
        """Revert an installed package.

        Args:
            package_id: Unique UUID of the package to rollback.

        Returns:
            ``True`` if rollback succeeded, ``False`` if package not found.
        """
        if package_id not in self.installed_packages:
            logger.warning("UpdateEngine: Rollback target %s not found", package_id)
            return False

        package = self.installed_packages[package_id]
        package.status = UpdateStatus.ROLLED_BACK
        del self.installed_packages[package_id]

        logger.info("UpdateEngine: Rolled back package %s v%s", package.name, package.version)
        return True

    async def list_installed(self) -> List[SkillPackage]:
        """List all currently operative skill packages.

        Returns:
            A list of installed ``SkillPackage`` objects.
        """
        return list(self.installed_packages.values())

    async def self_update(self) -> bool:
        """Auto-check specifically for NAYAK core evolution updates."""
        logger.info("UpdateEngine: Checking for NAYAK core updates...")
        updates = await self.check_updates()
        if updates:
            logger.info("UpdateEngine: %d updates are available for immediate deployment", len(updates))
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Register the updater and trigger a background self-scan."""
        await self.register()
        # Trigger background core update check
        asyncio.create_task(self.self_update())
        logger.info("UpdateEngine: Update engine initialized")

    async def stop(self) -> None:
        """Cleanly unregister the update engine."""
        try:
            registry.set_status(self.name, ModuleStatus.STOPPED)
            logger.info("UpdateEngine: Stopped")
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

updater: NayakUpdater = NayakUpdater()
"""Global NAYAK Update Engine instance."""

__all__ = ["NayakUpdater", "updater"]
