#!/usr/bin/env python3
"""Generate tiered Warzone gun loot tables with richer drops."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
GUN_TABLE_DIR = ROOT / "data" / "tacz" / "loot_tables" / "warzone" / "guns"
ALLOW_DIR = ROOT / "data" / "tacz" / "tags" / "attachments" / "allow_attachments"
ATTACHMENTS_DIR = ROOT / "data" / "tacz" / "attachments"
ATTACHMENT_TAG_DIR = ROOT / "data" / "tacz" / "tags" / "attachments"
LOADOUT_DIR = ROOT / "data" / "tacz" / "warzone" / "loadouts"

# Slot order to keep attachment NBT deterministic.
SLOT_PRIORITY = {
    "AttachmentSCOPE": 0,
    "AttachmentSIGHT": 1,
    "AttachmentMUZZLE": 2,
    "AttachmentEXTENDED_MAG": 3,
    "AttachmentAMMO_MOD": 4,
    "AttachmentSTOCK": 5,
    "AttachmentGRIP": 6,
    "AttachmentLASER": 7,
    "AttachmentBAYONET": 8,
}

COMMON_SLOTS = {
    "AttachmentSCOPE",
    "AttachmentSIGHT",
    "AttachmentMUZZLE",
    "AttachmentEXTENDED_MAG",
    "AttachmentSTOCK",
}

SLOT_DISPLAY = {
    "AttachmentSCOPE": "Scope",
    "AttachmentSIGHT": "Sight",
    "AttachmentMUZZLE": "Muzzle",
    "AttachmentEXTENDED_MAG": "Magazine",
    "AttachmentAMMO_MOD": "Ammo Mod",
    "AttachmentSTOCK": "Stock",
    "AttachmentGRIP": "Grip",
    "AttachmentLASER": "Laser",
    "AttachmentBAYONET": "Bayonet",
}

TIER_ORDER = ("common", "rare", "legendary")

TIER_META = {
    "common": {
        "display": "Common",
        "color": "gray",
        "weight": 9,
        "ammo_min_factor": 0.6,
        "ammo_max_factor": 0.85,
        "mag_drop_ratio": 0.25,
        "has_bullet": False,
        "cash_range": (1, 2),
        "armor_range": None,
        "killstreak": None,
    },
    "rare": {
        "display": "Rare",
        "color": "aqua",
        "weight": 5,
        "ammo_min_factor": 1.0,
        "ammo_max_factor": 1.0,
        "mag_drop_ratio": 0.0,
        "has_bullet": True,
        "cash_range": (2, 4),
        "armor_range": (1, 2),
        "killstreak": None,
    },
    "legendary": {
        "display": "Legendary",
        "color": "gold",
        "weight": 1,
        "ammo_min_factor": 1.25,
        "ammo_max_factor": 1.6,
        "mag_drop_ratio": -0.1,
        "has_bullet": True,
        "cash_range": (3, 6),
        "armor_range": (2, 3),
        "killstreak": 0.65,
    },
}

ATTACHMENT_TYPE_TO_SLOT = {
    "scope": "AttachmentSCOPE",
    "sight": "AttachmentSCOPE",
    "muzzle": "AttachmentMUZZLE",
    "silencer": "AttachmentMUZZLE",
    "grip": "AttachmentGRIP",
    "stock": "AttachmentSTOCK",
    "laser": "AttachmentLASER",
    "extended_mag": "AttachmentEXTENDED_MAG",
    "bayonet": "AttachmentBAYONET",
}

ATTACHMENT_ID_OVERRIDE_SLOT = {
    # Items that are expressed directly in allow tags without an attachment json (legacy stock presets)
    "tacz:oem_stock_heavy": "AttachmentSTOCK",
    "tacz:oem_stock_tactical": "AttachmentSTOCK",
    "tacz:oem_stock_light": "AttachmentSTOCK",
    "tacz:bayonet_6h3": "AttachmentBAYONET",
    "tacz:bayonet_m9": "AttachmentBAYONET",
}


@dataclass
class Loadout:
    gun_id: str
    fire_mode: str
    mag_current: int
    has_bullet: bool
    ammo_id: str
    ammo_min: int
    ammo_max: int
    base_attachments: Dict[str, str]


def load_jsonc(path: Path) -> Mapping[str, object]:
    text = path.read_text(encoding="utf-8")
    # Strip // comments.
    text = re.sub(r"//.*", "", text)
    # Remove trailing commas before closing braces/brackets.
    text = re.sub(r",(\s*[}\]])", r"\\1", text)
    return json.loads(text)


def parse_gun_table(path: Path) -> Loadout:
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = data["pools"][0]
    entry = pool["entries"][0]
    set_nbt = next(f for f in entry["functions"] if f["function"] == "minecraft:set_nbt")
    nbt = set_nbt["tag"]
    gun_id = _find_string(nbt, 'GunId:\\"([^\\"]+)\\"')
    fire_mode = _find_string(nbt, 'GunFireMode:\\"([^\\"]*)\\"')
    ammo_count = int(_find_number(nbt, r"GunCurrentAmmoCount:(\d+)") or 0)
    has_bullet = "HasBulletInBarrel:1b" in nbt
    attachments = {
        slot: att
        for slot, att in re.findall(
            '(Attachment[A-Z_]+):\\{id:\\"tacz:attachment\\",Count:1b,tag:\\{AttachmentId:\\"([^\\"]+)\\"',
            nbt,
        )
    }

    ammo_pool = data["pools"][1]
    ammo_entry = ammo_pool["entries"][0]
    ammo_nbt = next(f["tag"] for f in ammo_entry["functions"] if f["function"] == "minecraft:set_nbt")
    ammo_id = _find_string(ammo_nbt, 'AmmoId:\\"([^\\"]+)\\"')
    count_function = next(f for f in ammo_entry["functions"] if f["function"] == "minecraft:set_count")
    count = count_function["count"]
    if isinstance(count, Mapping):
        ammo_min = int(count["min"])
        ammo_max = int(count["max"])
    else:
        ammo_min = ammo_max = int(count)

    return Loadout(
        gun_id=gun_id,
        fire_mode=fire_mode,
        mag_current=ammo_count,
        has_bullet=has_bullet,
        ammo_id=ammo_id,
        ammo_min=ammo_min,
        ammo_max=ammo_max,
        base_attachments=dict(sorted(attachments.items(), key=lambda kv: SLOT_PRIORITY.get(kv[0], 100))),
    )


def _find_string(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _find_number(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def load_attachment_types() -> Dict[str, str]:
    types: Dict[str, str] = {}
    for path in ATTACHMENTS_DIR.glob("*.json"):
        data = load_jsonc(path)
        attachment_type = data.get("type")
        if isinstance(attachment_type, str):
            attachment_id = f"tacz:{path.stem}"
            types[attachment_id] = attachment_type
    return types


def load_attachment_tags() -> Dict[str, List[str]]:
    tags: Dict[str, List[str]] = {}
    for path in ATTACHMENT_TAG_DIR.glob("*.json"):
        if path.name == "allow_attachments":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tags[f"tacz:{path.stem}"] = list(data.get("values", []))
    return tags


class TagResolver:
    def __init__(self, tags: Mapping[str, Sequence[str]]):
        self._tags = tags
        self._cache: Dict[str, List[str]] = {}

    def resolve(self, tag_id: str) -> List[str]:
        if tag_id in self._cache:
            return list(self._cache[tag_id])
        values: List[str] = []
        for value in self._tags.get(tag_id, []):
            if value.startswith("#"):
                values.extend(self.resolve(value[1:]))
            else:
                values.append(value)
        self._cache[tag_id] = list(values)
        return values


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def slot_for_item(attachment_id: str, attachment_types: Mapping[str, str]) -> Optional[str]:
    if attachment_id in ATTACHMENT_ID_OVERRIDE_SLOT:
        return ATTACHMENT_ID_OVERRIDE_SLOT[attachment_id]
    attachment_type = attachment_types.get(attachment_id)
    if not attachment_type:
        return None
    return ATTACHMENT_TYPE_TO_SLOT.get(attachment_type, None)


def allowed_slots_for_gun(gun_name: str, attachment_types: Mapping[str, str], resolver: TagResolver) -> Dict[str, List[str]]:
    path = ALLOW_DIR / f"{gun_name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    slots: Dict[str, List[str]] = {}
    for value in data.get("values", []):
        attachments: List[str]
        if value.startswith("#"):
            attachments = resolver.resolve(value[1:])
        else:
            attachments = [value]
        for attachment_id in attachments:
            slot = slot_for_item(attachment_id, attachment_types)
            if not slot:
                continue
            slots.setdefault(slot, []).append(attachment_id)
    for slot in list(slots.keys()):
        slots[slot] = dedupe_preserve(slots[slot])
    return slots


def choose_lower(allowed: Sequence[str], rare_item: Optional[str]) -> Optional[str]:
    if not allowed:
        return rare_item
    for item in allowed:
        if rare_item is None or item != rare_item:
            return item
    return rare_item


def choose_higher(allowed: Sequence[str], rare_item: Optional[str]) -> Optional[str]:
    if not allowed:
        return rare_item
    for item in reversed(allowed):
        if rare_item is None or item != rare_item:
            return item
    return rare_item


def build_variant_attachments(
    loadout: Loadout,
    allowed_slots: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, str]]:
    rare = dict(loadout.base_attachments)
    common: Dict[str, str] = {}
    legendary: Dict[str, str] = {}

    for slot, allowed in allowed_slots.items():
        rare_item = rare.get(slot)
        if rare_item and rare_item not in allowed:
            allowed = list(allowed) + [rare_item]
        common_item = None
        if slot in COMMON_SLOTS:
            common_item = choose_lower(allowed, rare_item)
        if common_item:
            common[slot] = common_item
        elif slot in COMMON_SLOTS and rare_item:
            common[slot] = rare_item
        if rare_item:
            legendary_item = choose_higher(allowed, rare_item)
        else:
            legendary_item = choose_higher(allowed, None)
        if legendary_item:
            legendary[slot] = legendary_item

    # Ensure legendary keeps any base attachment even if allow data is missing.
    for slot, attachment in rare.items():
        legendary.setdefault(slot, attachment)
    # Preserve ordering.
    common = dict(sorted(common.items(), key=lambda kv: SLOT_PRIORITY.get(kv[0], 100)))
    rare = dict(sorted(rare.items(), key=lambda kv: SLOT_PRIORITY.get(kv[0], 100)))
    legendary = dict(sorted(legendary.items(), key=lambda kv: SLOT_PRIORITY.get(kv[0], 100)))
    return {"common": common, "rare": rare, "legendary": legendary}


def adjust_magazine(base: int, ratio: float) -> int:
    if base <= 1:
        return max(1, base)
    if ratio <= 0:
        return max(1, int(round(base * (1 - abs(ratio)))))
    drop = max(1, int(round(base * ratio)))
    return max(1, base - drop)


def boost_magazine(base: int, ratio: float) -> int:
    if ratio <= 0:
        return base
    bonus = max(1, int(round(base * ratio)))
    return base + bonus


def adjust_ammo_range(base_min: int, base_max: int, factor_min: float, factor_max: float) -> Tuple[int, int]:
    new_min = max(1, int(math.floor(base_min * factor_min)))
    new_max = max(new_min, int(math.ceil(base_max * factor_max)))
    return new_min, new_max


def format_display_name(gun_id: str) -> str:
    base = gun_id.split(":", 1)[1]
    base = base.replace("_", " ")
    base = re.sub(r"(?<=\D)(\d)", r"-\1", base)
    return base.upper()


def format_attachment_name(attachment_id: str) -> str:
    name = attachment_id.split(":", 1)[1]
    name = name.replace("_", " ")
    return name.title()


def format_ammo_id(ammo_id: str) -> str:
    return ammo_id.split(":", 1)[1].replace("_", " ")


def build_display_nbt(name_json: Mapping[str, object], lore: Sequence[Mapping[str, object]]) -> str:
    name_str = json.dumps(name_json, separators=(",", ":"))
    lore_entries = ",".join("'" + json.dumps(entry, separators=(",", ":")) + "'" for entry in lore)
    return f",display:{{Name:'{name_str}',Lore:[{lore_entries}]}}"


def build_attachment_nbt(attachments: Mapping[str, str]) -> str:
    pieces = []
    for slot, attachment in attachments.items():
        pieces.append(
            f",{slot}:{{id:\"tacz:attachment\",Count:1b,tag:{{AttachmentId:\"{attachment}\"}}}}"
        )
    return "".join(pieces)


def build_gun_nbt(
    loadout: Loadout,
    attachments: Mapping[str, str],
    tier: str,
    tier_meta: Mapping[str, object],
    ammo_count: int,
    name_json: Mapping[str, object],
    lore: Sequence[Mapping[str, object]],
    has_bullet: bool,
) -> str:
    display_nbt = build_display_nbt(name_json, lore)
    attachment_nbt = build_attachment_nbt(attachments)
    gun_fire = loadout.fire_mode
    return (
        "{"
        f"GunId:\"{loadout.gun_id}\""
        f",GunFireMode:\"{gun_fire}\""
        f",GunCurrentAmmoCount:{ammo_count}"
        f",HasBulletInBarrel:{'1b' if has_bullet else '0b'}"
        f",WarzoneRarity:\"{tier}\""
        f"{display_nbt}{attachment_nbt}"
        "}"
    )


def build_lore(
    tier: str,
    tier_meta: Mapping[str, object],
    attachments: Mapping[str, str],
    ammo_range: Tuple[int, int],
    ammo_id: str,
) -> List[Mapping[str, object]]:
    lore: List[Mapping[str, object]] = [
        {"text": "Warzone Drop", "color": tier_meta["color"], "italic": False},
        {"text": f"Tier: {tier_meta['display']}", "color": tier_meta["color"], "italic": False},
        {
            "text": f"Ammo: {ammo_range[0]}-{ammo_range[1]} {format_ammo_id(ammo_id)}",
            "color": "white",
            "italic": False,
        },
    ]
    if attachments:
        lore.append({"text": "Attachments:", "color": "gray", "italic": False})
        for slot, attachment in attachments.items():
            label = SLOT_DISPLAY.get(slot, slot.replace("Attachment", ""))
            lore.append(
                {
                    "text": f" - {label}: {format_attachment_name(attachment)}",
                    "color": "dark_gray",
                    "italic": False,
                }
            )
    else:
        lore.append({"text": "Attachments: None", "color": "dark_gray", "italic": False})
    return lore


def build_item_entry(name: str, count: Tuple[int, int] | int, nbt_tag: Optional[str], chance: Optional[float] = None) -> Mapping[str, object]:
    entry: Dict[str, object] = {"type": "minecraft:item", "name": name}
    if chance is not None:
        entry["conditions"] = [
            {"condition": "minecraft:random_chance", "chance": chance}
        ]
    functions: List[Mapping[str, object]] = []
    if isinstance(count, tuple):
        if count[0] == count[1]:
            functions.append({"function": "minecraft:set_count", "count": count[0]})
        else:
            functions.append(
                {
                    "function": "minecraft:set_count",
                    "count": {"min": count[0], "max": count[1]},
                }
            )
    else:
        functions.append({"function": "minecraft:set_count", "count": count})
    if nbt_tag:
        functions.append({"function": "minecraft:set_nbt", "tag": nbt_tag})
    entry["functions"] = functions
    return entry


def build_cash_nbt() -> str:
    display = {
        "Name": {"text": "Cash Bundle", "color": "green", "italic": False},
        "Lore": [
            {"text": "$500", "color": "green", "italic": False},
            {"text": "Spend at buy stations", "color": "dark_green", "italic": False},
        ],
    }
    return encode_display(display)


def build_armor_nbt() -> str:
    display = {
        "Name": {"text": "Armor Plate", "color": "white", "italic": False},
        "Lore": [
            {"text": "Restores armor", "color": "gray", "italic": False}
        ],
    }
    return encode_display(display)


def build_killstreak_nbt() -> str:
    display = {
        "Name": {"text": "Killstreak Token", "color": "gold", "italic": False},
        "Lore": [
            {"text": "Deploy for devastation", "color": "yellow", "italic": False}
        ],
    }
    return encode_display(display)


def encode_display(display: Mapping[str, object]) -> str:
    lore = display.get("Lore", [])
    lore_entries = ",".join("'" + json.dumps(entry, separators=(",", ":")) + "'" for entry in lore)
    name_json = json.dumps(display.get("Name", {"text": ""}), separators=(",", ":"))
    if lore_entries:
        return f"{{display:{{Name:'{name_json}',Lore:[{lore_entries}]}}}}"
    return f"{{display:{{Name:'{name_json}'}}}}"


def build_extra_items(tier: str, tier_meta: Mapping[str, object]) -> List[Mapping[str, object]]:
    extras: List[Mapping[str, object]] = []
    cash_range = tier_meta.get("cash_range")
    if cash_range:
        extras.append(
            build_item_entry(
                "minecraft:emerald",
                (int(cash_range[0]), int(cash_range[1])),
                build_cash_nbt(),
            )
        )
    armor_range = tier_meta.get("armor_range")
    if armor_range:
        extras.append(
            build_item_entry(
                "minecraft:iron_ingot",
                (int(armor_range[0]), int(armor_range[1])),
                build_armor_nbt(),
                chance=0.75 if tier == "rare" else None,
            )
        )
    killstreak_chance = tier_meta.get("killstreak")
    if killstreak_chance:
        extras.append(
            build_item_entry(
                "minecraft:nether_star",
                1,
                build_killstreak_nbt(),
                chance=float(killstreak_chance),
            )
        )
    return extras


def build_variant_table(
    loadout: Loadout,
    tier: str,
    tier_meta: Mapping[str, object],
    attachments: Mapping[str, str],
) -> Mapping[str, object]:
    ammo_range = adjust_ammo_range(
        loadout.ammo_min,
        loadout.ammo_max,
        tier_meta["ammo_min_factor"],
        tier_meta["ammo_max_factor"],
    )
    if tier == "legendary" and tier_meta.get("mag_drop_ratio", 0) < 0:
        mag_count = boost_magazine(loadout.mag_current, abs(float(tier_meta["mag_drop_ratio"])) )
    elif tier == "common":
        mag_count = adjust_magazine(loadout.mag_current, float(tier_meta["mag_drop_ratio"]))
    else:
        mag_count = loadout.mag_current
    has_bullet = bool(tier_meta.get("has_bullet", False))

    name_json = {
        "text": f"{format_display_name(loadout.gun_id)} | {tier_meta['display'].upper()}",
        "color": tier_meta["color"],
        "italic": False,
    }
    lore = build_lore(tier, tier_meta, attachments, ammo_range, loadout.ammo_id)
    gun_nbt = build_gun_nbt(loadout, attachments, tier, tier_meta, mag_count, name_json, lore, has_bullet)

    gun_entry = {
        "type": "minecraft:item",
        "name": "tacz:modern_kinetic_gun",
        "functions": [
            {"function": "minecraft:set_count", "count": 1},
            {"function": "minecraft:set_nbt", "tag": gun_nbt},
        ],
    }
    ammo_entry = {
        "type": "minecraft:item",
        "name": "tacz:ammo",
        "functions": [
            {"function": "minecraft:set_nbt", "tag": f"{{AmmoId:\"{loadout.ammo_id}\"}}"},
            {
                "function": "minecraft:set_count",
                "count": {"min": ammo_range[0], "max": ammo_range[1]},
            },
        ],
    }
    extras = build_extra_items(tier, tier_meta)
    sequence_children: List[Mapping[str, object]] = [gun_entry, ammo_entry]
    sequence_children.extend(extras)
    return {
        "type": "minecraft:chest",
        "pools": [
            {
                "rolls": 1,
                "bonus_rolls": 0,
                "entries": [
                    {
                        "type": "minecraft:sequence",
                        "children": sequence_children,
                    }
                ],
            }
        ],
    }


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_loadout(path: Path, loadout: Loadout) -> None:
    payload = {
        "gun_id": loadout.gun_id,
        "fire_mode": loadout.fire_mode,
        "mag_current": loadout.mag_current,
        "has_bullet": loadout.has_bullet,
        "ammo": {
            "id": loadout.ammo_id,
            "min": loadout.ammo_min,
            "max": loadout.ammo_max,
        },
        "base_attachments": loadout.base_attachments,
    }
    write_json(path, payload)


def load_loadout(gun_name: str, gun_path: Path) -> Loadout:
    loadout_path = LOADOUT_DIR / f"{gun_name}.json"
    if loadout_path.exists():
        data = json.loads(loadout_path.read_text(encoding="utf-8"))
        return Loadout(
            gun_id=data["gun_id"],
            fire_mode=data.get("fire_mode", ""),
            mag_current=int(data.get("mag_current", 0)),
            has_bullet=bool(data.get("has_bullet", False)),
            ammo_id=data["ammo"]["id"],
            ammo_min=int(data["ammo"]["min"]),
            ammo_max=int(data["ammo"]["max"]),
            base_attachments={str(k): str(v) for k, v in data.get("base_attachments", {}).items()},
        )
    loadout = parse_gun_table(gun_path)
    write_loadout(loadout_path, loadout)
    return loadout


def build_variant_tables() -> None:
    attachment_types = load_attachment_types()
    resolver = TagResolver(load_attachment_tags())

    gun_paths = {path.stem: path for path in GUN_TABLE_DIR.glob("*.json")}
    gun_names = sorted(set(gun_paths.keys()) | {path.stem for path in LOADOUT_DIR.glob("*.json")})

    for tier in TIER_ORDER:
        tier_dir = GUN_TABLE_DIR / tier
        if tier_dir.exists():
            for existing in tier_dir.glob("*.json"):
                existing.unlink()
        tier_dir.mkdir(parents=True, exist_ok=True)

    per_gun_tables: Dict[str, Dict[str, str]] = {}

    for gun_name in gun_names:
        gun_path = gun_paths.get(gun_name, GUN_TABLE_DIR / f"{gun_name}.json")
        loadout = load_loadout(gun_name, gun_path)
        allowed = allowed_slots_for_gun(gun_name, attachment_types, resolver)
        attachments = build_variant_attachments(loadout, allowed)

        per_gun_tables[gun_name] = {}
        for tier in TIER_ORDER:
            table = build_variant_table(loadout, tier, TIER_META[tier], attachments[tier])
            tier_path = GUN_TABLE_DIR / tier / f"{gun_name}.json"
            write_json(tier_path, table)
            per_gun_tables[gun_name][tier] = f"tacz:warzone/guns/{tier}/{gun_name}"

        # Write aggregator for this gun
        aggregator_entries = [
            {
                "type": "minecraft:loot_table",
                "name": per_gun_tables[gun_name][tier],
                "weight": TIER_META[tier]["weight"],
            }
            for tier in TIER_ORDER
        ]
        aggregator = {
            "type": "minecraft:chest",
            "pools": [
                {
                    "rolls": 1,
                    "bonus_rolls": 0,
                    "entries": aggregator_entries,
                }
            ],
        }
        write_json(GUN_TABLE_DIR / f"{gun_name}.json", aggregator)

    write_master_tables(gun_names, per_gun_tables)
    write_setup_function(gun_names, per_gun_tables)


def write_master_tables(gun_names: Sequence[str], per_gun_tables: Mapping[str, Mapping[str, str]]) -> None:
    # Standard chest referencing per-gun aggregators
    chest_entries = [
        {"type": "minecraft:loot_table", "name": f"tacz:warzone/guns/{gun_name}", "weight": 1}
        for gun_name in gun_names
    ]
    chest = {
        "type": "minecraft:chest",
        "pools": [
            {"rolls": 1, "bonus_rolls": 0, "entries": chest_entries}
        ],
    }
    write_json(ROOT / "data" / "tacz" / "loot_tables" / "warzone" / "chest.json", chest)

    for tier in TIER_ORDER:
        entries = [
            {
                "type": "minecraft:loot_table",
                "name": per_gun_tables[gun][tier],
                "weight": 1,
            }
            for gun in gun_names
        ]
        table = {
            "type": "minecraft:chest",
            "pools": [
                {"rolls": 1, "bonus_rolls": 0, "entries": entries}
            ],
        }
        write_json(
            ROOT
            / "data"
            / "tacz"
            / "loot_tables"
            / "warzone"
            / f"chest_{tier}.json",
            table,
        )


def write_setup_function(gun_names: Sequence[str], per_gun_tables: Mapping[str, Mapping[str, str]]) -> None:
    lines = [
        "# Auto-generated by tools/refresh_warzone_loot.py",
        "data modify storage tacz:warzone guns set value []",
        "data modify storage tacz:warzone guns_common set value []",
        "data modify storage tacz:warzone guns_rare set value []",
        "data modify storage tacz:warzone guns_legendary set value []",
        "data modify storage tacz:warzone tier_weights set value {common:%d,rare:%d,legendary:%d}" % (
            TIER_META["common"]["weight"],
            TIER_META["rare"]["weight"],
            TIER_META["legendary"]["weight"],
        ),
    ]
    for gun in gun_names:
        lines.append(f'data modify storage tacz:warzone guns append value "tacz:warzone/guns/{gun}"')
        for tier in TIER_ORDER:
            lines.append(
                f'data modify storage tacz:warzone guns_{tier} append value "{per_gun_tables[gun][tier]}"'
            )
    path = ROOT / "data" / "tacz" / "functions" / "warzone" / "setup.mcfunction"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_variant_tables()
    print("Refreshed Warzone loot tables with tiered loadouts.")
