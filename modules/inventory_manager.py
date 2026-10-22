"""
Smart Farm Inventory Management Module.
Implements OOP architecture, custom exceptions, schema validation, and dual CSV/JSON persistence.
Tracks Fertilizers, Pesticides, Seeds, and Farm Supplies.
"""

import os
import json
import csv
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Union

class InventoryException(Exception):
    """Base exception for inventory operations."""
    pass

class ItemNotFoundError(InventoryException):
    """Raised when an inventory item is not found."""
    def __init__(self, item_id: str):
        super().__init__(f"Inventory Item '{item_id}' was not found in database.")
        self.item_id = item_id

class InsufficientStockError(InventoryException):
    """Raised when attempting to deduct more quantity than available in stock."""
    def __init__(self, item_id: str, requested: float, available: float):
        super().__init__(
            f"Insufficient stock for '{item_id}': Requested {requested}, but only {available} available."
        )
        self.item_id = item_id
        self.requested = requested
        self.available = available

class InvalidItemDataError(InventoryException):
    """Raised when item fields fail validation."""
    def __init__(self, message: str, invalid_field: Optional[str] = None):
        super().__init__(f"Invalid item data: {message}")
        self.invalid_field = invalid_field

class StorageCorruptedError(InventoryException):
    """Raised when CSV or JSON file is corrupt or unreadable."""
    def __init__(self, file_path: str, reason: str):
        super().__init__(f"Storage corruption in '{file_path}': {reason}")
        self.file_path = file_path

VALID_CATEGORIES = ["Fertilizer", "Pesticide", "Seeds", "Equipment", "Other"]

@dataclass
class InventoryItem:
    item_id: str
    name: str
    category: str
    quantity: float
    unit: str
    min_threshold: float
    price_per_unit: float
    location: str
    expiry_date: str
    supplier: str = "Unknown"
    notes: str = ""

    def __post_init__(self):
        if self.category not in VALID_CATEGORIES:
            raise InvalidItemDataError(
                f"Category '{self.category}' is invalid. Allowed: {', '.join(VALID_CATEGORIES)}",
                invalid_field="category"
            )
        try:
            self.quantity = float(self.quantity)
            if self.quantity < 0:
                raise ValueError("Quantity cannot be negative.")
        except Exception as e:
            raise InvalidItemDataError(f"Invalid quantity: {e}", invalid_field="quantity")

        try:
            self.min_threshold = float(self.min_threshold)
            if self.min_threshold < 0:
                raise ValueError("Threshold cannot be negative.")
        except Exception as e:
            raise InvalidItemDataError(f"Invalid min_threshold: {e}", invalid_field="min_threshold")

        try:
            self.price_per_unit = float(self.price_per_unit)
            if self.price_per_unit < 0:
                raise ValueError("Price cannot be negative.")
        except Exception as e:
            raise InvalidItemDataError(f"Invalid price_per_unit: {e}", invalid_field="price_per_unit")

    @property
    def total_value(self) -> float:
        """Calculates total monetary value of current stock."""
        return round(self.quantity * self.price_per_unit, 2)

    @property
    def is_low_stock(self) -> bool:
        """Determines if stock level is at or below safe threshold."""
        return self.quantity <= self.min_threshold

    @property
    def is_out_of_stock(self) -> bool:
        """Determines if item has zero inventory."""
        return self.quantity <= 0.001

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Calculates days until item reaches expiration date."""
        try:
            exp = datetime.strptime(self.expiry_date, "%Y-%m-%d").date()
            delta = (exp - date.today()).days
            return delta
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Converts item instance to dictionary."""
        d = asdict(self)
        d["total_value"] = self.total_value
        d["is_low_stock"] = self.is_low_stock
        d["days_until_expiry"] = self.days_until_expiry
        return d

class InventoryManager:
    """
    Core OOP Inventory Controller.
    Manages agricultural inventory with dual JSON/CSV synchronization,
    transaction logging, and robust exception recovery.
    """

    def __init__(self, json_path: str = None, csv_path: str = None, log_path: str = None):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.json_path = json_path or os.path.join(base_dir, "inventory.json")
        self.csv_path = csv_path or os.path.join(base_dir, "inventory.csv")
        self.log_path = log_path or os.path.join(base_dir, "inventory_audit.log")

        self.items: Dict[str, InventoryItem] = {}
        self.load_inventory()

    def load_inventory(self) -> None:
        """
        Loads inventory from JSON with fallback to CSV or auto-initialization.
        Handles corruption, missing files, and type conversion.
        """
        self.items = {}
        loaded = False

        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for entry in raw_data:
                        try:
                            valid_keys = InventoryItem.__dataclass_fields__.keys()
                            filtered = {k: v for k, v in entry.items() if k in valid_keys}
                            item = InventoryItem(**filtered)
                            self.items[item.item_id] = item
                        except Exception as parse_err:
                            print(f"[Warning] Skipped malformed JSON item: {parse_err}")
                    loaded = True
            except Exception as e:
                print(f"[Warning] Failed to load JSON inventory: {e}. Trying CSV fallback.")

        if not loaded and os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    valid_keys = InventoryItem.__dataclass_fields__.keys()
                    for row in reader:
                        try:
                            filtered = {k: row[k] for k in valid_keys if k in row}
                            item = InventoryItem(**filtered)
                            self.items[item.item_id] = item
                        except Exception as row_err:
                            print(f"[Warning] Skipped malformed CSV row: {row_err}")
                loaded = True
            except Exception as e:
                raise StorageCorruptedError(self.csv_path, f"CSV parsing failed: {e}")

        if not self.items:
            print("[Info] Starting with empty inventory.")

    def save_inventory(self) -> None:
        """
        Saves current state simultaneously to both JSON and CSV files
        for redundant data integrity.
        """
        raw_items = [asdict(item) for item in self.items.values()]

        try:
            temp_json = self.json_path + ".tmp"
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, indent=2)
            if os.path.exists(self.json_path):
                os.replace(temp_json, self.json_path)
            else:
                os.rename(temp_json, self.json_path)
        except Exception as e:
            raise StorageCorruptedError(self.json_path, f"Failed writing JSON: {e}")

        try:
            temp_csv = self.csv_path + ".tmp"
            fieldnames = list(InventoryItem.__dataclass_fields__.keys())
            with open(temp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in self.items.values():
                    writer.writerow(asdict(item))
            if os.path.exists(self.csv_path):
                os.replace(temp_csv, self.csv_path)
            else:
                os.rename(temp_csv, self.csv_path)
        except Exception as e:
            raise StorageCorruptedError(self.csv_path, f"Failed writing CSV: {e}")

    def log_action(self, action: str, item_id: str, details: str) -> None:
        """Records an audit trail event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] ACTION={action.upper()} | ITEM={item_id} | {details}\n"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"[Warning] Could not append to audit log: {e}")

    def get_all_items(self) -> List[InventoryItem]:
        """Returns all items currently stored."""
        return list(self.items.values())

    def get_item(self, item_id: str) -> InventoryItem:
        """Fetches single item by ID or raises ItemNotFoundError."""
        if item_id not in self.items:
            raise ItemNotFoundError(item_id)
        return self.items[item_id]

    def add_item(self, item: Union[InventoryItem, dict]) -> InventoryItem:
        """
        Adds a new inventory item with validation.
        Prevents duplicate primary keys.
        """
        if isinstance(item, dict):
            item = InventoryItem(**item)

        if not item.item_id or not item.name:
            raise InvalidItemDataError("Item ID and Name must not be empty.")

        if item.item_id in self.items:
            raise InvalidItemDataError(f"Item with ID '{item.item_id}' already exists. Use update instead.")

        self.items[item.item_id] = item
        self.save_inventory()
        self.log_action("ADD", item.item_id, f"Added '{item.name}' ({item.quantity} {item.unit})")
        return item

    def update_item(self, item_id: str, updates: Dict[str, Any]) -> InventoryItem:
        """Updates specific fields of an existing item."""
        item = self.get_item(item_id)
        current_data = asdict(item)

        for key, value in updates.items():
            if key in current_data and key != "item_id":
                current_data[key] = value

        updated_item = InventoryItem(**current_data)
        self.items[item_id] = updated_item
        self.save_inventory()
        self.log_action("UPDATE", item_id, f"Updated fields: {list(updates.keys())}")
        return updated_item

    def use_stock(self, item_id: str, amount_used: float, purpose: str = "") -> InventoryItem:
        """
        Deducts stock for farm application (fertilizing, pest spraying, sowing).
        Enforces non-negative constraints with InsufficientStockError.
        """
        if amount_used <= 0:
            raise InvalidItemDataError("Stock deduction amount must be strictly greater than zero.")

        item = self.get_item(item_id)
        if item.quantity < amount_used:
            raise InsufficientStockError(item_id, requested=amount_used, available=item.quantity)

        new_qty = round(item.quantity - amount_used, 2)
        item.quantity = new_qty
        self.save_inventory()
        self.log_action("USE", item_id, f"Deducted {amount_used} {item.unit}. Remaining: {new_qty}. Note: {purpose}")
        return item

    def restock(self, item_id: str, amount_added: float, new_price: Optional[float] = None) -> InventoryItem:
        """Replenishes item stock with optional price adjustment."""
        if amount_added <= 0:
            raise InvalidItemDataError("Restock amount must be greater than zero.")

        item = self.get_item(item_id)
        item.quantity = round(item.quantity + amount_added, 2)
        if new_price is not None and new_price > 0:
            item.price_per_unit = round(new_price, 2)

        self.save_inventory()
        self.log_action("RESTOCK", item_id, f"Added {amount_added} {item.unit}. Total: {item.quantity}")
        return item

    def delete_item(self, item_id: str) -> bool:
        """Removes an item from the inventory catalog."""
        item = self.get_item(item_id)
        del self.items[item_id]
        self.save_inventory()
        self.log_action("DELETE", item_id, f"Removed '{item.name}' from inventory.")
        return True

    def get_low_stock_items(self) -> List[InventoryItem]:
        """Returns items where quantity is at or below minimum threshold."""
        return [item for item in self.items.values() if item.is_low_stock]

    def get_expiring_items(self, within_days: int = 90) -> List[InventoryItem]:
        """Returns items approaching expiration within given number of days."""
        expiring = []
        for item in self.items.values():
            days = item.days_until_expiry
            if days is not None and 0 <= days <= within_days:
                expiring.append(item)
        return expiring

    def get_total_valuation(self) -> float:
        """Computes aggregate monetary value of all items in stock."""
        return round(sum(item.total_value for item in self.items.values()), 2)

    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """Aggregates inventory counts, units, and valuation grouped by category."""
        summary = {}
        for cat in VALID_CATEGORIES:
            cat_items = [i for i in self.items.values() if i.category == cat]
            summary[cat] = {
                "count": len(cat_items),
                "total_value": round(sum(i.total_value for i in cat_items), 2),
                "low_stock_count": sum(1 for i in cat_items if i.is_low_stock),
                "items": cat_items
            }
        return summary
