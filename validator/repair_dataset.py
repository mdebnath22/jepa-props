"""
Self-Supervised Corruption Dataset for JEPA Repair Pretraining
===============================================================
Generates (broken_ref, xml_context, correct_ref) triples from valid MuJoCo XMLs
by applying the SAME corruption patterns that real LLMs produce (from the paper's
taxonomy):

  R1a: capitalisation mismatch    hip_FR → hip_fr
  R1b: hyphen/underscore swap     knee-joint → knee_joint
  R1c: truncated suffix           knee_joint_1 → knee_jt_1
  R1d: partial artifact           hip_fr → joint_xyz_829   (garbled)

This is the self-supervised training signal: corrupt a valid XML,
then train JEPA to predict the original (correct) symbol from context.

No labels needed — just a corpus of valid MJCF files.
"""

import re
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from validator.jepa_repair_model import tokenize_symbol, tokenize_context


# ─────────────────────────────────────────────────────────────────────────────
# Corruption functions (mirroring real LLM error patterns from the paper)
# ─────────────────────────────────────────────────────────────────────────────

def corrupt_capitalisation(name: str, rng: random.Random) -> str:
    """R1a: randomly change case of some characters."""
    chars = list(name)
    n_flips = rng.randint(1, max(1, len(name) // 3))
    alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
    if not alpha_indices:
        return name
    flip_indices = rng.sample(alpha_indices, min(n_flips, len(alpha_indices)))
    for i in flip_indices:
        chars[i] = chars[i].lower() if chars[i].isupper() else chars[i].upper()
    return "".join(chars)


def corrupt_hyphen_underscore(name: str, rng: random.Random) -> str:
    """R1b: swap hyphens and underscores."""
    result = ""
    for c in name:
        if c == "_" and rng.random() < 0.7:
            result += "-"
        elif c == "-" and rng.random() < 0.7:
            result += "_"
        else:
            result += c
    return result


def corrupt_truncated_suffix(name: str, rng: random.Random) -> str:
    """R1c: truncate or abbreviate part of the name."""
    parts = re.split(r"([_\-])", name)  # split keeping delimiters
    if len(parts) <= 1:
        # No delimiter — truncate last few chars
        trunc_len = rng.randint(1, max(1, len(name) // 3))
        return name[:-trunc_len] if trunc_len < len(name) else name[:2]
    # Abbreviate one of the word parts
    word_indices = [i for i in range(len(parts)) if parts[i] not in ("_", "-")]
    if not word_indices:
        return name
    idx = rng.choice(word_indices)
    word = parts[idx]
    if len(word) > 2:
        abbrev_len = rng.randint(1, len(word) - 1)
        parts[idx] = word[:abbrev_len]
    return "".join(parts)


def corrupt_garbled(name: str, rng: random.Random) -> str:
    """R1d: generate a plausible-looking but semantically wrong identifier."""
    prefixes = ["joint", "motor", "link", "body", "hinge", "slide", "geom"]
    suffix = rng.randint(100, 999)
    return f"{rng.choice(prefixes)}_xyz_{suffix}"


CORRUPTION_FNS = [
    (corrupt_capitalisation,   0.40),   # most common per paper
    (corrupt_hyphen_underscore, 0.25),
    (corrupt_truncated_suffix,  0.25),
    (corrupt_garbled,           0.10),   # R1d — rare, hard to fix
]


def corrupt_name(name: str, rng: random.Random) -> str:
    """Apply a weighted-random corruption to a joint name."""
    r = rng.random()
    cumulative = 0.0
    for fn, weight in CORRUPTION_FNS:
        cumulative += weight
        if r < cumulative:
            corrupted = fn(name, rng)
            # Make sure we actually changed it
            if corrupted != name:
                return corrupted
    # Fallback
    return corrupt_capitalisation(name, rng)


# ─────────────────────────────────────────────────────────────────────────────
# XML context extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_joint_names(xml_str: str) -> List[str]:
    """Extract all joint names declared in <worldbody>."""
    names = []
    try:
        root = ET.fromstring(xml_str)
        for body in root.iter("body"):
            for joint in body.findall("joint"):
                name = joint.get("name")
                if name:
                    names.append(name)
    except ET.ParseError:
        # Regex fallback for malformed XML
        names = re.findall(r'<joint[^>]+name=["\']([^"\']+)["\']', xml_str)
    return names


def extract_motor_references(xml_str: str) -> List[Tuple[str, int, int]]:
    """
    Extract (joint_ref, start_pos, end_pos) for each motor joint= attribute.
    Returns list of (ref_name, char_start, char_end) in the original string.
    """
    refs = []
    for m in re.finditer(r'<motor[^>]+joint=["\']([^"\']+)["\']', xml_str):
        refs.append((m.group(1), m.start(1), m.end(1)))
    return refs


def get_context_window(xml_str: str, pos: int, window: int = 200) -> str:
    """
    Extract a context window around position pos in the XML string.
    Includes both before and after context.
    """
    start = max(0, pos - window // 2)
    end = min(len(xml_str), pos + window // 2)
    return xml_str[start:end]


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation from a single valid XML
# ─────────────────────────────────────────────────────────────────────────────

def generate_repair_samples_from_xml(
    xml_str: str,
    rng: random.Random,
    n_samples_per_xml: int = 5,
    context_window: int = 256,
) -> List[Dict]:
    """
    Given a valid MJCF XML string, generate (broken, context, correct) triples.

    Each sample:
        {
          'broken_ref'   : str  — corrupted joint name
          'correct_ref'  : str  — original joint name
          'xml_context'  : str  — surrounding XML snippet
          'all_declared' : list — all declared joint names
          'corruption_type': str
        }
    """
    joint_names = extract_joint_names(xml_str)
    if not joint_names:
        return []

    motor_refs = extract_motor_references(xml_str)
    if not motor_refs:
        return []

    samples = []
    for _ in range(n_samples_per_xml):
        # Pick a random motor reference to corrupt
        ref_name, ref_start, ref_end = rng.choice(motor_refs)

        # Only corrupt if this reference is actually a declared joint name
        if ref_name not in joint_names:
            continue

        # Determine corruption type for labeling
        r = rng.random()
        cumulative = 0.0
        corruption_type = "capitalisation"
        for fn, weight in CORRUPTION_FNS:
            cumulative += weight
            if r < cumulative:
                corruption_type = fn.__name__
                break

        broken = corrupt_name(ref_name, rng)
        if broken == ref_name:
            continue  # failed to corrupt, skip

        # Extract XML context around the motor reference
        context = get_context_window(xml_str, ref_start, window=context_window)

        # Replace the joint name in the context with the corrupted version
        # (so the model sees the broken reference in context)
        broken_context = context.replace(ref_name, broken, 1)

        samples.append({
            "broken_ref":      broken,
            "correct_ref":     ref_name,
            "xml_context":     broken_context,
            "all_declared":    joint_names,
            "corruption_type": corruption_type,
        })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic XML generator (for when we don't have a corpus)
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAME_TEMPLATES = [
    "hip_{side}_{leg}",
    "knee_{side}_{leg}",
    "ankle_{side}_{leg}",
    "shoulder_{side}",
    "elbow_{side}",
    "spine_{n}",
    "tail_{n}",
    "fin_{n}_{side}",
    "hip_joint_{n}",
    "knee_joint_{n}",
]

def generate_synthetic_mjcf(
    n_joints: int = 6,
    task: str = "locomotion",
    rng: random.Random = None,
) -> str:
    """
    Generate a minimal but structurally valid MJCF string for pretraining.
    Produces joint declarations in worldbody and motor references in actuator.
    """
    rng = rng or random.Random()
    sides = ["FR", "FL", "RR", "RL"]
    legs = ["thigh", "shin"]

    joint_names = []
    for i in range(n_joints):
        template = rng.choice(JOINT_NAME_TEMPLATES)
        name = template.format(
            side=rng.choice(sides),
            leg=rng.choice(legs),
            n=i,
        )
        # Ensure uniqueness
        if name in joint_names:
            name = f"{name}_{i}"
        joint_names.append(name)

    # Build worldbody section
    bodies = []
    for i, jname in enumerate(joint_names):
        body = f"""
    <body name="body_{i}" pos="0 0 {0.1*i}">
      <joint name="{jname}" type="hinge" axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="0.04 0.1" rgba="0.8 0.6 0.4 1"/>
    </body>"""
        bodies.append(body)

    # Build actuator section (correct references)
    motors = []
    for jname in joint_names:
        gear = rng.randint(30, 80)
        motors.append(f'    <motor joint="{jname}" gear="{gear}"/>')

    option = ""
    if task == "swimming":
        option = '<option viscosity="0.8" density="1000" gravity="0 0 0"/>'

    bodies_str = "".join(bodies)
    motors_str = "\n".join(motors)
    xml = f"""<mujoco model="synthetic_{task}">
  {option}
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="torso" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.1"/>
      {bodies_str}
    </body>
  </worldbody>
  <actuator>
{motors_str}
  </actuator>
</mujoco>"""
    return xml


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class JEPARepairDataset(Dataset):
    """
    Dataset of (broken_ref, xml_context, correct_ref) triples for
    JEPA repair model pretraining.

    Can be built from:
      - A list of valid XML strings (real corpus)
      - Synthetic generation (no corpus needed)
    """

    def __init__(
        self,
        samples: List[Dict],
        max_symbol_len: int = 64,
        max_context_len: int = 256,
    ):
        self.samples = samples
        self.max_symbol_len = max_symbol_len
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "broken_ids":  tokenize_symbol(s["broken_ref"],  self.max_symbol_len),
            "correct_ids": tokenize_symbol(s["correct_ref"], self.max_symbol_len),
            "context_ids": tokenize_context(s["xml_context"], self.max_context_len),
            "corruption_type": s.get("corruption_type", "unknown"),
        }

    @classmethod
    def from_xml_strings(
        cls,
        xml_strings: List[str],
        n_samples_per_xml: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        rng = random.Random(seed)
        all_samples = []
        for xml_str in xml_strings:
            samples = generate_repair_samples_from_xml(xml_str, rng, n_samples_per_xml)
            all_samples.extend(samples)
        return cls(all_samples, **kwargs)

    @classmethod
    def from_synthetic(
        cls,
        n_xmls: int = 500,
        joints_range: Tuple[int, int] = (3, 16),
        n_samples_per_xml: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        """Build dataset from synthetically generated MJCF strings."""
        rng = random.Random(seed)
        tasks = ["locomotion", "jumping", "swimming", "crawling", "pushing"]
        xml_strings = []
        for i in range(n_xmls):
            n_joints = rng.randint(*joints_range)
            task = rng.choice(tasks)
            xml_str = generate_synthetic_mjcf(n_joints=n_joints, task=task, rng=rng)
            xml_strings.append(xml_str)
        return cls.from_xml_strings(xml_strings, n_samples_per_xml, seed, **kwargs)
