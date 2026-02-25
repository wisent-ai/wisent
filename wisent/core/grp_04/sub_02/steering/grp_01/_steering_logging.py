"""Response logging mixin for SteeringMethod."""
import datetime
import json
import os
from typing import Any, Dict, List, Optional

from wisent.core.constants import JSON_INDENT


class SteeringLoggingMixin:
    """Mixin providing response logging methods."""

    def enable_response_logging(self, log_file_path: str = "./harmful_responses.json") -> None:
        """
        Enable logging of harmful responses.

        Args:
            log_file_path: Path to the log file
        """
        self.enable_logging = True
        self.log_file_path = log_file_path

        # Initialize log file if it doesn't exist
        if not os.path.exists(os.path.dirname(log_file_path)):
            try:
                os.makedirs(os.path.dirname(log_file_path))
            except Exception:
                pass

        if not os.path.exists(log_file_path):
            try:
                with open(log_file_path, "w") as f:
                    json.dump([], f)
            except Exception:
                pass

    def log_harmful_response(
        self, prompt: str, response: str, probability: float, category: str = "harmful", additional_info: Dict = None
    ) -> bool:
        """
        Log a harmful response to the JSON log file.

        Args:
            prompt: The original prompt
            response: The generated response
            probability: The probability score that triggered detection
            category: The category of harmful content detected
            additional_info: Optional additional information

        Returns:
            Success flag
        """
        if not self.enable_logging:
            return False

        try:
            # Create log entry
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "probability": float(probability),
                "category": category,
                "threshold": float(self.threshold),
                "method_type": self.method_type.value,
            }

            # Add additional info if provided
            if additional_info:
                log_entry.update(additional_info)

            # Read existing log entries
            try:
                with open(self.log_file_path) as f:
                    log_entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log_entries = []

            # Append new entry
            log_entries.append(log_entry)

            # Write updated log
            with open(self.log_file_path, "w") as f:
                json.dump(log_entries, f, indent=JSON_INDENT)

            return True

        except Exception:
            return False

    def get_logged_responses(self, limit: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve logged harmful responses from the log file.

        Args:
            limit: Maximum number of entries to return (None for all)
            category: Filter by specific category (None for all categories)

        Returns:
            List of log entries
        """
        if not self.enable_logging:
            return []

        try:
            # Check if log file exists
            if not os.path.exists(self.log_file_path):
                return []

            # Read log entries
            with open(self.log_file_path) as f:
                log_entries = json.load(f)

            # Filter by category if specified
            if category is not None:
                log_entries = [entry for entry in log_entries if entry.get("category") == category]

            # Sort by timestamp (newest first)
            log_entries.sort(key=lambda entry: entry.get("timestamp", ""), reverse=True)

            # Apply limit if specified
            if limit is not None and limit > 0:
                log_entries = log_entries[:limit]

            return log_entries

        except Exception:
            return []

