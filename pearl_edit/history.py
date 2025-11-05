"""History management for undo/redo functionality."""
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from .state import SessionState, ImageRecord

logger = logging.getLogger(__name__)


class HistoryError(Exception):
    """Raised when history operations fail."""
    pass


def serialize_state(state: SessionState, temp_dir: Path) -> Dict[str, Any]:
    """
    Serialize session state to a dictionary.
    
    Args:
        state: SessionState to serialize
        temp_dir: Temp directory path for making paths relative
        
    Returns:
        Dictionary representation of state
    """
    try:
        # Make paths relative to temp_dir
        def make_relative(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                return str(path.relative_to(temp_dir))
            except ValueError:
                # If path is not relative, store as absolute
                return str(path)
        
        images_data = []
        for img in state.images:
            images_data.append({
                'image_index': img.image_index,
                'original_image': make_relative(img.original_image),
                'split_image': make_relative(img.split_image),
                'left_or_right': img.left_or_right
            })
        
        return {
            'current_image_index': state.current_image_index,
            'status': state.status,
            'images': images_data
        }
    except Exception as e:
        logger.error(f"Error serializing state: {e}")
        raise HistoryError(f"Failed to serialize state: {str(e)}") from e


def restore_state(state: SessionState, data: Dict[str, Any], temp_dir: Path) -> None:
    """
    Restore session state from a dictionary.
    
    Args:
        state: SessionState to restore into
        data: Dictionary representation of state
        temp_dir: Temp directory path for resolving relative paths
    """
    try:
        # Resolve paths relative to temp_dir
        def resolve_path(path_str: Optional[str]) -> Optional[Path]:
            if path_str is None:
                return None
            path = Path(path_str)
            if path.is_absolute():
                return path
            return temp_dir / path
        
        state.current_image_index = data['current_image_index']
        state.status = data.get('status', 'no_changes')
        
        state.images.clear()
        for img_data in data['images']:
            record = ImageRecord(
                image_index=img_data['image_index'],
                original_image=resolve_path(img_data['original_image']),
                split_image=resolve_path(img_data['split_image']),
                left_or_right=img_data.get('left_or_right')
            )
            state.images.append(record)
    except Exception as e:
        logger.error(f"Error restoring state: {e}")
        raise HistoryError(f"Failed to restore state: {str(e)}") from e


class HistoryManager:
    """Manages undo/redo history using file snapshots."""
    
    def __init__(self, temp_dir: Optional[Path]):
        """
        Initialize history manager.
        
        Args:
            temp_dir: Base temp directory for storing history
        """
        self.temp_dir = temp_dir
        self.history_dir: Optional[Path] = None
        if temp_dir:
            self.history_dir = temp_dir / "history"
            self.history_dir.mkdir(exist_ok=True)
        
        self.undo_stack: List[str] = []  # List of operation IDs
        self.redo_stack: List[str] = []  # List of operation IDs
        self._current_op_id: Optional[str] = None
        self._op_counter = 0
    
    def _get_next_op_id(self, op_name: str) -> str:
        """Generate next operation ID."""
        self._op_counter += 1
        # Sanitize op_name for filesystem
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in op_name)
        return f"op_{self._op_counter:06d}_{safe_name}"
    
    def _get_op_dir(self, op_id: str) -> Path:
        """Get directory for an operation."""
        if not self.history_dir:
            raise HistoryError("History directory not initialized")
        return self.history_dir / op_id
    
    def start(self, op_name: str, affected_paths: List[Path], session_state: SessionState) -> str:
        """
        Start a new history operation.
        
        Args:
            op_name: Name/description of the operation
            affected_paths: List of file paths that will be modified or deleted
            session_state: Current session state to snapshot
            
        Returns:
            Operation ID for this operation
        """
        if not self.history_dir:
            logger.warning("History directory not available, skipping history")
            return ""
        
        op_id = self._get_next_op_id(op_name)
        op_dir = self._get_op_dir(op_id)
        op_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session state
        state_data = serialize_state(session_state, self.temp_dir)
        state_file = op_dir / "state_pre.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Copy affected files
        files_dir = op_dir / "files"
        files_dir.mkdir(exist_ok=True)
        
        for file_path in affected_paths:
            if file_path.exists():
                # Store relative path for portability
                rel_path = file_path.name  # Use filename for simplicity
                backup_path = files_dir / rel_path
                try:
                    shutil.copy2(file_path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to backup {file_path}: {e}")
        
        self._current_op_id = op_id
        logger.debug(f"Started history operation: {op_id}")
        return op_id
    
    def commit(self, op_id: str, created_paths: List[Path] = None) -> None:
        """
        Commit a history operation.
        
        Args:
            op_id: Operation ID returned by start()
            created_paths: List of file paths that were created by this operation
        """
        if not op_id or not self.history_dir:
            return
        
        op_dir = self._get_op_dir(op_id)
        if not op_dir.exists():
            logger.warning(f"Operation directory not found: {op_dir}")
            return
        
        # Save list of created files
        if created_paths:
            created_data = {
                'files': [str(p) for p in created_paths]
            }
            created_file = op_dir / "created_files.json"
            with open(created_file, 'w') as f:
                json.dump(created_data, f, indent=2)
        
        # Add to undo stack
        self.undo_stack.append(op_id)
        
        # Clear redo stack when new operation is committed
        self.redo_stack.clear()
        
        self._current_op_id = None
        logger.debug(f"Committed history operation: {op_id}")
    
    def undo(self, session_state: SessionState) -> bool:
        """
        Undo the last operation.
        
        Args:
            session_state: Current session state to restore
            
        Returns:
            True if undo was successful
        """
        if not self.undo_stack:
            return False
        
        # Get the operation to undo
        op_id = self.undo_stack.pop()
        op_dir = self._get_op_dir(op_id)
        
        if not op_dir.exists():
            logger.error(f"Operation directory not found: {op_id}")
            return False
        
        try:
            # Snapshot current state for redo
            current_op_id = self._get_next_op_id("redo")
            current_op_dir = self._get_op_dir(current_op_id)
            current_op_dir.mkdir(parents=True, exist_ok=True)
            
            # Save current state
            current_state_data = serialize_state(session_state, self.temp_dir)
            current_state_file = current_op_dir / "state_pre.json"
            with open(current_state_file, 'w') as f:
                json.dump(current_state_data, f, indent=2)
            
            # Read created files list from the operation we're undoing
            created_files = []
            created_file = op_dir / "created_files.json"
            if created_file.exists():
                with open(created_file, 'r') as f:
                    created_data = json.load(f)
                    created_files = [Path(p) for p in created_data.get('files', [])]
            
            # Save current state files for redo
            files_dir = current_op_dir / "files"
            files_dir.mkdir(exist_ok=True)
            
            # Save all current image files
            for record in session_state.images:
                try:
                    if record.current_image_path.exists():
                        shutil.copy2(record.current_image_path, files_dir / record.current_image_path.name)
                except Exception:
                    pass
                try:
                    if record.original_image.exists() and record.original_image != record.current_image_path:
                        shutil.copy2(record.original_image, files_dir / record.original_image.name)
                except Exception:
                    pass
            
            # Save current created files for redo (they'll be deleted, so we need to save them)
            created_files_dir = current_op_dir / "created_files"
            created_files_dir.mkdir(exist_ok=True)
            for created_path in created_files:
                if created_path.exists():
                    try:
                        shutil.copy2(created_path, created_files_dir / created_path.name)
                    except Exception as e:
                        logger.warning(f"Failed to backup created file {created_path}: {e}")
            
            # Save created files list for redo (already saved the files above)
            if created_files:
                created_data = {
                    'files': [str(p) for p in created_files]
                }
                created_file = current_op_dir / "created_files.json"
                with open(created_file, 'w') as f:
                    json.dump(created_data, f, indent=2)
            
            # Read state to know which files need to be restored
            state_file = op_dir / "state_pre.json"
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Delete created files FIRST to avoid filename collisions
            # This is critical when re-splitting, as the left split might overwrite
            # the old split image file with the same name
            for created_path in created_files:
                if created_path.exists():
                    try:
                        created_path.unlink()
                        logger.debug(f"Deleted created file for undo: {created_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete created file {created_path}: {e}")
            
            # Restore files from backup, ensuring they exist before restoring state
            # This ensures files exist when state references them
            restore_files_dir = op_dir / "files"
            if restore_files_dir.exists():
                # Build a map of backup files by name for quick lookup
                backup_files = {}
                for backup_file in restore_files_dir.iterdir():
                    if backup_file.is_file():
                        backup_files[backup_file.name] = backup_file
                
                # Restore files that are referenced in the state
                # This ensures we restore to the exact paths needed
                def resolve_path(path_str: Optional[str]) -> Optional[Path]:
                    if path_str is None:
                        return None
                    path = Path(path_str)
                    if path.is_absolute():
                        return path
                    return self.temp_dir / path
                
                # Restore all files that might be referenced
                for img_data in state_data.get('images', []):
                    # Restore original_image if it exists in backup
                    orig_path = resolve_path(img_data.get('original_image'))
                    if orig_path and orig_path.name in backup_files:
                        try:
                            # Ensure parent directory exists
                            orig_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_files[orig_path.name], orig_path)
                            logger.debug(f"Restored original image: {orig_path}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {orig_path.name}: {e}")
                    
                    # Restore split_image if it exists in backup
                    split_path = resolve_path(img_data.get('split_image'))
                    if split_path and split_path.name in backup_files:
                        try:
                            # Ensure parent directory exists
                            split_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_files[split_path.name], split_path)
                            logger.debug(f"Restored split image: {split_path}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {split_path.name}: {e}")
                
                # Also restore any other files in the backup that might be needed
                for backup_name, backup_file in backup_files.items():
                    target_file = self.temp_dir / backup_name
                    # Only restore if not already restored above
                    if not target_file.exists():
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_file, target_file)
                            logger.debug(f"Restored additional file: {target_file}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {backup_name}: {e}")
            
            # Now restore state (files should exist now)
            restore_state(session_state, state_data, self.temp_dir)
            
            # Add to redo stack
            self.redo_stack.append(current_op_id)
            
            logger.debug(f"Undone operation: {op_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during undo: {e}", exc_info=True)
            # Put operation back on stack
            self.undo_stack.append(op_id)
            return False
    
    def redo(self, session_state: SessionState) -> bool:
        """
        Redo the last undone operation.
        
        Args:
            session_state: Current session state to restore
            
        Returns:
            True if redo was successful
        """
        if not self.redo_stack:
            return False
        
        # Get the operation to redo
        op_id = self.redo_stack.pop()
        op_dir = self._get_op_dir(op_id)
        
        if not op_dir.exists():
            logger.error(f"Operation directory not found: {op_id}")
            return False
        
        try:
            # Snapshot current state for undo
            current_op_id = self._get_next_op_id("undo")
            current_op_dir = self._get_op_dir(current_op_id)
            current_op_dir.mkdir(parents=True, exist_ok=True)
            
            # Save current state
            current_state_data = serialize_state(session_state, self.temp_dir)
            current_state_file = current_op_dir / "state_pre.json"
            with open(current_state_file, 'w') as f:
                json.dump(current_state_data, f, indent=2)
            
            # Copy current files
            files_dir = current_op_dir / "files"
            files_dir.mkdir(exist_ok=True)
            
            for record in session_state.images:
                try:
                    if record.current_image_path.exists():
                        shutil.copy2(record.current_image_path, files_dir / record.current_image_path.name)
                except Exception:
                    pass
            
            # Read created files list
            created_files = []
            created_file = op_dir / "created_files.json"
            if created_file.exists():
                with open(created_file, 'r') as f:
                    created_data = json.load(f)
                    created_files = [Path(p) for p in created_data.get('files', [])]
            
            # Read state to know which files need to be restored
            state_file = op_dir / "state_pre.json"
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore files FIRST, before restoring state
            # This ensures files exist when state references them
            restore_files_dir = op_dir / "files"
            if restore_files_dir.exists():
                # Build a map of backup files by name for quick lookup
                backup_files = {}
                for backup_file in restore_files_dir.iterdir():
                    if backup_file.is_file():
                        backup_files[backup_file.name] = backup_file
                
                # Restore files that are referenced in the state
                def resolve_path(path_str: Optional[str]) -> Optional[Path]:
                    if path_str is None:
                        return None
                    path = Path(path_str)
                    if path.is_absolute():
                        return path
                    return self.temp_dir / path
                
                # Restore all files that might be referenced
                for img_data in state_data.get('images', []):
                    # Restore original_image if it exists in backup
                    orig_path = resolve_path(img_data.get('original_image'))
                    if orig_path and orig_path.name in backup_files:
                        try:
                            # Ensure parent directory exists
                            orig_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_files[orig_path.name], orig_path)
                            logger.debug(f"Restored original image for redo: {orig_path}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {orig_path.name}: {e}")
                    
                    # Restore split_image if it exists in backup
                    split_path = resolve_path(img_data.get('split_image'))
                    if split_path and split_path.name in backup_files:
                        try:
                            # Ensure parent directory exists
                            split_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_files[split_path.name], split_path)
                            logger.debug(f"Restored split image for redo: {split_path}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {split_path.name}: {e}")
                
                # Also restore any other files in the backup that might be needed
                for backup_name, backup_file in backup_files.items():
                    target_file = self.temp_dir / backup_name
                    # Only restore if not already restored above
                    if not target_file.exists():
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup_file, target_file)
                            logger.debug(f"Restored additional file for redo: {target_file}")
                        except Exception as e:
                            logger.warning(f"Failed to restore {backup_name}: {e}")
            
            # Restore created files (these will overwrite any conflicting restored files, which is correct for redo)
            created_files_dir = op_dir / "created_files"
            if created_files_dir.exists():
                for created_backup in created_files_dir.iterdir():
                    if created_backup.is_file():
                        # Find the corresponding path from created_files list
                        for created_path in created_files:
                            if created_path.name == created_backup.name:
                                try:
                                    # Ensure parent directory exists
                                    created_path.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(created_backup, created_path)
                                    logger.debug(f"Restored created file for redo: {created_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to restore created file {created_path}: {e}")
                                break
            
            # Now restore state (files should exist now)
            restore_state(session_state, state_data, self.temp_dir)
            
            # Add to undo stack
            self.undo_stack.append(current_op_id)
            
            logger.debug(f"Redone operation: {op_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during redo: {e}", exc_info=True)
            # Put operation back on stack
            self.redo_stack.append(op_id)
            return False
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
    
    def cleanup(self) -> None:
        """Clean up history directory."""
        if self.history_dir and self.history_dir.exists():
            try:
                shutil.rmtree(self.history_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup history: {e}")

