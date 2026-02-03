# -*- coding: utf-8 -*-
import json, re, os, collections, shutil, time
from typing import Dict, Any, List, Tuple, Optional

# ---------- parse INIT facts ----------
_fact_re = re.compile(r"\(\s*([^\s()]+)\s*([^\)]*?)\)")

def parse_facts(init_text: str):
    facts = []
    for m in _fact_re.finditer(init_text):
        pred = m.group(1)
        args = [a for a in m.group(2).split() if a]
        facts.append((pred, args))
    return facts

def base_name(s: str):  # shortest alias: before "_bar_"
    return s.split("_bar_")[0]

def make_aliases(names):
    buckets = collections.defaultdict(list)
    for n in sorted(names):
        buckets[base_name(n)].append(n)
    alias = {}
    for b, lst in buckets.items():
        for i, n in enumerate(lst, 1):
            alias[n] = f"{b.lower()}{i}"
    return alias


# ---------- helpers: read blocks ----------
def extract_block(text: str, head="init"):
    start = text.find(f"(:{head}")
    if start < 0: return ""
    i, depth = start, 0
    while i < len(text):
        c = text[i]
        if c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start + len(f"(:{head}"): i].strip()
        i += 1
    return ""


# ---------- INIT summary (English, compact) ----------
def summarize_init_english(problem_text: str):
    init_text = extract_block(problem_text, "init")
    facts = parse_facts(init_text)

    obj2otype, rec2rtype = {}, {}
    otypes, rtypes = set(), set()
    canContain = set()

    CAPS = {
        "pickupable": set(),
        "toggleable": set(),
        "cleanable": set(),
        "heatable": set(),
        "coolable": set(),
        "sliceable": set(),
    }

    locs = set()
    for p, a in facts:
        if p == "objectType" and len(a) == 2:
            obj2otype[a[0]] = a[1]; otypes.add(a[1])
        elif p == "receptacleType" and len(a) == 2:
            rec2rtype[a[0]] = a[1]; rtypes.add(a[1])
        elif p == "canContain" and len(a) == 2:
            canContain.add(tuple(a))
        elif p in CAPS and len(a) == 1:
            CAPS[p].add(a[0])
        elif p in ("receptacleAtLocation", "objectAtLocation") and len(a) == 2:
            locs.add(a[1])
        elif p == "atLocation" and len(a) == 2:
            locs.add(a[1])

    obj_alias = make_aliases(obj2otype.keys())
    rec_alias = make_aliases(rec2rtype.keys())
    loc_alias = make_aliases(locs)

    otype2objs = collections.defaultdict(list)
    for o, t in obj2otype.items():
        otype2objs[t].append(obj_alias[o])
    for t in otype2objs:
        otype2objs[t].sort(key=lambda s: (re.sub(r"\d+$","",s), int(re.search(r"(\d+)$", s).group(1))))

    rtype2recs = collections.defaultdict(list)
    for r, t in rec2rtype.items():
        rtype2recs[t].append(rec_alias[r])
    for t in rtype2recs:
        rtype2recs[t].sort(key=lambda s: (re.sub(r"\d+$","",s), int(re.search(r"(\d+)$", s).group(1))))

    unary = {
        "pickupable","openable","opened","isReceptacleObject","isReceptacleObjectFull",
        "cleanable","heatable","coolable","full","moveable","toggleable","isOn","isToggled",
        "sliceable","isSliced","checked","holdsAny","holdsAnyReceptacleObject"
    }
    groups = collections.defaultdict(list)
    for p, a in facts:
        if p in ("objectType","receptacleType","canContain"): continue
        if p in unary and len(a) == 1:
            tok = a[0]
            tok = obj_alias.get(tok, rec_alias.get(tok, loc_alias.get(tok, tok)))
            groups[p].append(tok)
        else:
            tup = tuple(obj_alias.get(x, rec_alias.get(x, loc_alias.get(x, x))) for x in a)
            groups[p].append(tup)

    for k, v in list(groups.items()):
        if not v: continue
        if isinstance(v[0], tuple):
            groups[k] = sorted(set(v))
        else:
            groups[k] = sorted(set(v))

    EXPLAIN = {
        "_otype": "types → concrete objects",
        "_rtype": "types → concrete receptacles",
        "_locs": "list of location aliases",
        "_cancontain": "receptacle-class can contain object-class",
        "receptacleAtLocation": "receptacle r is at location l",
        "objectAtLocation": "object o is at location l",
        "atLocation": "agent is at location l",
        "inReceptacle": "object o is in/on receptacle r",
        "inReceptacleObject": "object inner is inside object outer",
        "wasInReceptacle": "object o was/is in receptacle r (historical)",
        "openable": "receptacle can be opened",
        "opened": "receptacle is currently open",
        "pickupable": "object can be picked up by the agent",
        "isReceptacleObject": "object can itself receive other objects",
        "isReceptacleObjectFull": "receptacle-object is currently full",
        "cleanable": "object can be cleaned in a sink",
        "heatable": "object can be heated in a microwave",
        "coolable": "object can be cooled in a fridge",
        "toggleable": "object has an on/off state",
        "isOn": "object is currently on",
        "isToggled": "object has been toggled",
        "sliceable": "object can be sliced",
        "isSliced": "object is sliced",
        "holds": "agent holds object o",
        "holdsAny": "agent holds something",
        "holdsAnyReceptacleObject": "agent holds a receptacle-object",
        "full": "receptacle is full",
        "moveable": "object can be moved",
        "checked": "entity has been examined",
    }

    out = []
    out.append(f"### Object classes (otype) → instances — {EXPLAIN['_otype']}")
    if otype2objs:
        for t in sorted(otype2objs):
            out.append(f"- {t}: " + ", ".join(otype2objs[t]))
    else:
        out.append("- (no objectType facts)")

    out.append(f"\n### Receptacle classes (rtype) → instances — {EXPLAIN['_rtype']}")
    if rtype2recs:
        for t in sorted(rtype2recs):
            out.append(f"- {t}: " + ", ".join(rtype2recs[t]))
    else:
        out.append("- (no receptacleType facts)")

    out.append(f"\n### Locations — {EXPLAIN['_locs']}")
    out.append(", ".join([loc_alias[k] for k in sorted(loc_alias)]) if loc_alias else "(none)")

    out.append(f"\n### canContain (class → class) — {EXPLAIN['_cancontain']}")
    out.append(", ".join(f"({rt}, {ot})" for rt, ot in sorted(canContain)) if canContain else "(none)")

    preferred = [
        "receptacleAtLocation","objectAtLocation","atLocation",
        "inReceptacle","inReceptacleObject","wasInReceptacle",
        "openable","opened","pickupable","isReceptacleObject","isReceptacleObjectFull",
        "cleanable","heatable","coolable","toggleable","isOn","isToggled",
        "sliceable","isSliced","holds","holdsAny","holdsAnyReceptacleObject",
        "full","moveable","checked"
    ]
    keys = [k for k in preferred if k in groups] + [k for k in sorted(groups) if k not in preferred]
    for k in keys:
        vals = groups[k]
        if not vals: continue
        explain = EXPLAIN.get(k, "other instantiated facts")
        out.append(f"\n### {k} — {explain}")
        if isinstance(vals[0], tuple):
            out.append(", ".join("(" + ", ".join(x for x in tup) + ")" for tup in vals))
        else:
            out.append(", ".join(vals))

    return "\n".join(out), {
        "obj2otype": obj2otype, "rec2rtype": rec2rtype,
        "otype2objs": otype2objs, "rtype2recs": rtype2recs,
        "canContain": canContain, "caps": CAPS
    }

# ---------- detect task type from path ----------
TASK_NAMES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}
def detect_task_type_from_path(path: str):
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if "-" in p:
            first = p.split("-")[0]
            if first in TASK_NAMES:
                return first
    if len(parts) >= 2 and "-" in parts[-2]:
        first = parts[-2].split("-")[0]
        if first in TASK_NAMES: return first
    return "unknown"

# ---------- GOAL token handling ----------
_re_objtype = re.compile(r"\(\s*objectType\s+[^\s)]+\s+([A-Za-z0-9]+Type)\s*\)", re.I)
_re_rectype = re.compile(r"\(\s*receptacleType\s+[^\s)]+\s+([A-Za-z0-9]+Type)\s*\)", re.I)

def goal_detect_types(goal_raw: str):
    def uniq(seq):
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return uniq(_re_objtype.findall(goal_raw)), uniq(_re_rectype.findall(goal_raw))

def goal_mask_types_consistent(goal_raw: str):
    obj_map, rec_map = {}, {}
    def repl_obj(m):
        t = m.group(1)
        if t not in obj_map:
            obj_map[t] = f"{{OBJ_TYPE_{len(obj_map)+1}}}"
        return m.group(0).replace(t, obj_map[t])
    def repl_rec(m):
        t = m.group(1)
        if t not in rec_map:
            rec_map[t] = f"{{REC_TYPE_{len(rec_map)+1}}}"
        return m.group(0).replace(t, rec_map[t])
    tmp = _re_objtype.sub(repl_obj, goal_raw)
    tmp = _re_rectype.sub(repl_rec, tmp)
    return tmp, obj_map, rec_map

# ---------- small validators ----------
def can_contain_ok(rt, ot, canContain_set): return (rt, ot) in canContain_set
def has_capability_type(ot, cap_set, obj2otype):
    return any(o in cap_set for o, t in obj2otype.items() if t == ot)
def count_instances_of_type(ot, obj2otype):
    return sum(1 for _, t in obj2otype.items() if t == ot)

# ---------- task-specific goal brief + design instruction ----------
def goal_brief_and_instruction(task, goal_obj_types, goal_rec_types, S, direction: Optional[str] = None, prev_acc: Optional[float] = None):
    lines = []
    if direction in ("harder", "easier"):
        lines.append(f"### Difficulty goal: **{direction.upper()}** (prev_acc={prev_acc})")
        if direction == "harder":
            lines.append("- Prefer types with *fewer* available instances (rarer) while keeping constraints satisfied.")
            lines.append("- Prefer combinations likely requiring more search/steps, but still solvable in this environment.")
        else:
            lines.append("- Prefer types with *more* available instances (more common) while keeping constraints satisfied.")
            lines.append("- Prefer combinations likely easier to find/complete, but still valid.")

    if task == "pick_and_place_simple":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_rec_types[0] if goal_rec_types else "<?>")
        lines.append("**Overall Framework**: place an object type into/on a receptacle type.")
        lines.append(f"**Original goal**: place an object of type **{g[0]}** into/on a receptacle of type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("**Design instruction**: Output exactly **two tokens** — `<OBJ_TYPE> <REC_TYPE>`.")  # your parse uses comma anyway
        lines.append("- Constraints: pair must satisfy `canContain(REC_TYPE, OBJ_TYPE)`.")
    elif task == "look_at_obj_in_light":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_obj_types[1] if len(goal_obj_types)>1 else "<?>")
        lines.append("**Overall Framework**: a light object type is present at the agent’s location; the agent **holds** an object type.")
        lines.append(f"**Original goal (Example)**: a **toggleable and toggled** light object of type **{g[0]}** is present; agent **holds** type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("**Design instruction**: Output exactly **two tokens** — `<LIGHT_OBJ_TYPE> <HOLD_OBJ_TYPE>`.")  # parse as two tokens
        lines.append("- Constraints: LIGHT must have `toggleable`; HOLD should have `pickupable` instance in INIT.")
    elif task == "pick_clean_then_place_in_recep":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_rec_types[0] if goal_rec_types else "<?>")
        lines.append("**Overall Framework**: **clean** an object type and place it into/on a receptacle type.")
        lines.append(f"**Original goal**: clean type **{g[0]}** then place into/on type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("- Constraints: OBJ must be cleanable; canContain(REC,OBJ).")
    elif task == "pick_heat_then_place_in_recep":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_rec_types[0] if goal_rec_types else "<?>")
        lines.append("**Overall Framework**: **heat** an object type and place it into/on a receptacle type.")
        lines.append(f"**Original goal**: heat type **{g[0]}** then place into/on type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("- Constraints: OBJ must be heatable; canContain(REC,OBJ).")
    elif task == "pick_cool_then_place_in_recep":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_rec_types[0] if goal_rec_types else "<?>")
        lines.append("**Overall Framework**: **cool** an object type and place it into/on a receptacle type.")
        lines.append(f"**Original goal**: cool type **{g[0]}** then place into/on type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("- Constraints: OBJ must be coolable; canContain(REC,OBJ).")
    elif task == "pick_two_obj_and_place":
        g = (goal_obj_types[0] if goal_obj_types else "<?>", goal_rec_types[0] if goal_rec_types else "<?>")
        lines.append("**Overall Framework**: place **two distinct objects** (same type) into/on a receptacle type.")
        lines.append(f"**Original goal**: place two objects of type **{g[0]}** into/on type **{g[1]}**.")
        lines.append(f"The final output example is \\boxed{{{g[0]}, {g[1]}}}")
        lines.append("- Constraints: >=2 instances of OBJ_TYPE; canContain(REC,OBJ).")
    else:
        lines.append("**Original goal**: (unknown task type).")
    return "\n".join(lines)


# —— type mapping helpers ——
def _norm_type(t: str) -> str:
    t = (t or "").strip()
    return t if t.endswith("Type") else t + "Type"

def decide_new_types(task, goal_obj_types, goal_rec_types, type1, type2, S=None):
    t1, t2 = _norm_type(type1), _norm_type(type2)

    if len(goal_rec_types) == 1 and len(goal_obj_types) == 1:
        new_types = {"OBJ_TYPE_1": t1, "REC_TYPE_1": t2}
        if S is not None and ("canContain" in S):
            if (new_types["REC_TYPE_1"], new_types["OBJ_TYPE_1"]) not in S["canContain"]:
                swapped = {"OBJ_TYPE_1": t2, "REC_TYPE_1": t1}
                if (swapped["REC_TYPE_1"], swapped["OBJ_TYPE_1"]) in S["canContain"]:
                    new_types = swapped
        return new_types

    if len(goal_rec_types) == 0 and len(goal_obj_types) == 2:
        return {"OBJ_TYPE_1": t1, "OBJ_TYPE_2": t2}

    raise ValueError(f"Unsupported goal signature: obj={goal_obj_types}, rec={goal_rec_types}")

def type_to_short(t):  # "BookType" -> "Book"
    return t[:-4] if t.endswith("Type") else t

def collect_placeholders(goal_template: str):
    return sorted(set(re.findall(r"\{(OBJ_TYPE_\d+|REC_TYPE_\d+)\}", goal_template)))

def fill_goal_from_template(goal_template: str, mapping: dict):
    filled = goal_template
    phs = collect_placeholders(goal_template)
    missing = [p for p in phs if p not in mapping]
    if missing:
        print("[WARN] Missing tokens for:", ", ".join(missing))
    for ph in phs:
        filled = filled.replace("{"+ph+"}", mapping.get(ph, f"<MISSING_{ph}>"))
    return filled

def replace_types_direct(goal_raw: str, old_obj_types, old_rec_types, mapping: dict):
    filled = goal_raw
    for i, ot in enumerate(old_obj_types, 1):
        key = f"OBJ_TYPE_{i}"
        if key in mapping:
            filled = re.sub(rf"\b{re.escape(ot)}\b", mapping[key], filled)
    for i, rt in enumerate(old_rec_types, 1):
        key = f"REC_TYPE_{i}"
        if key in mapping:
            filled = re.sub(rf"\b{re.escape(rt)}\b", mapping[key], filled)
    return filled

# ====== path canonicalization helpers ======
def canonical_game_id(game_path: str, env_data_dir: str) -> str:
    dataset_root = os.path.realpath(os.path.join(env_data_dir, "json_2.1.1"))
    p = os.path.realpath(game_path)
    if p.startswith(dataset_root + os.sep):
        rel = os.path.relpath(p, dataset_root)
        return rel.replace(os.sep, "/")
    return p

def resolve_game_id(game_id: str, env_data_dir: str) -> str:
    dataset_root = os.path.join(env_data_dir, "json_2.1.1")
    if os.path.isabs(game_id):
        return game_id
    return os.path.join(dataset_root, game_id.replace("/", os.sep))

# ====== suffix stripping to avoid explosion ======
_syn_suffix_re = re.compile(r"(?:-\d+-syn-\d+)+$")

def strip_syn_suffix(name: str) -> str:
    return _syn_suffix_re.sub("", name or "")

def build_name_with_replacements(orig_name, short_repls, step, node_index):
    orig_name = strip_syn_suffix(orig_name)
    parts = orig_name.split("-")
    new_parts = []
    for p in parts:
        new_p = p
        for old_short, new_short in short_repls:
            if new_p == old_short:
                new_p = new_short
            else:
                new_p = re.sub(rf"(?<![A-Za-z]){re.escape(old_short)}(?![A-Za-z])", new_short, new_p)
        new_parts.append(new_p)
    candidate = "-".join(new_parts)
    candidate += f"-{node_index}-syn-{step}"
    return candidate

def synth_task_names_from_path(
    game_path: str,
    new_types: dict,
    goal_obj_types,
    goal_rec_types,
    dest_split,
    project,
    step,
    node_index,
    attempt_id: Optional[int] = None,
):
    trial_dir = os.path.dirname(game_path)
    task_dir  = os.path.dirname(trial_dir)
    split_dir = os.path.dirname(task_dir)
    dataset_root = os.path.dirname(split_dir)  # .../json_2.1.1

    dest_root = os.path.join(dataset_root, project, dest_split)
    os.makedirs(dest_root, exist_ok=True)

    orig_task_folder  = strip_syn_suffix(os.path.basename(task_dir))
    orig_trial_folder = strip_syn_suffix(os.path.basename(trial_dir))

    short_repls = []
    for i, ot in enumerate(goal_obj_types, 1):
        key = f"OBJ_TYPE_{i}"
        if key in new_types:
            short_repls.append((type_to_short(ot), type_to_short(new_types[key])))
    for i, rt in enumerate(goal_rec_types, 1):
        key = f"REC_TYPE_{i}"
        if key in new_types:
            short_repls.append((type_to_short(rt), type_to_short(new_types[key])))

    new_task_folder  = build_name_with_replacements(orig_task_folder, short_repls, step, node_index)
    #new_trial_folder = orig_trial_folder + f"-{node_index}-syn-{step}"
    attempt_tag = f"-try{int(attempt_id)}" if attempt_id is not None else ""
    new_trial_folder = orig_trial_folder + attempt_tag + f"-{node_index}-syn-{step}"

    new_task_dir  = os.path.join(dest_root, new_task_folder)
    new_trial_dir = os.path.join(new_task_dir, new_trial_folder)
    return new_task_dir, new_trial_dir

def extract_block_raw(text: str, head="goal"):
    start = text.find(f"(:{head}")
    if start < 0: return ""
    i, depth = start, 0
    while i < len(text):
        c = text[i]
        if c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
        i += 1
    return ""

def substitute_goal_in_problem(problem_text: str, new_goal_block: str):
    old_goal = extract_block_raw(problem_text, "goal")
    if not old_goal:
        raise RuntimeError("No (:goal ...) block found in pddl_problem.")
    return problem_text.replace(old_goal, new_goal_block)

def validate_selection(task, S, new_types):
    obj2otype, canContain, caps = S["obj2otype"], S["canContain"], S["caps"]
    def ok_can(rt, ot): return (rt, ot) in canContain
    def has_cap(cap, ot):
        return any(o in caps[cap] for o, t in obj2otype.items() if t == ot)
    def count_type(ot): return sum(1 for _, t in obj2otype.items() if t == ot)

    if task in {"pick_and_place_simple","pick_clean_then_place_in_recep","pick_heat_then_place_in_recep","pick_cool_then_place_in_recep"}:
        ot = new_types.get("OBJ_TYPE_1")
        rt = new_types.get("REC_TYPE_1")
        if ot and rt and not ok_can(rt, ot):
            print(f"[WARN] canContain({rt}, {ot}) not present in INIT.")
        req = {
            "pick_clean_then_place_in_recep": "cleanable",
            "pick_heat_then_place_in_recep": "heatable",
            "pick_cool_then_place_in_recep": "coolable",
        }.get(task)
        if req and ot and not has_cap(req, ot):
            print(f"[WARN] No instance of {ot} marked `{req}` in INIT.")
    elif task == "look_at_obj_in_light":
        lt = new_types.get("OBJ_TYPE_1")
        ht = new_types.get("OBJ_TYPE_2")
        if lt and not has_cap("toggleable", lt):
            print(f"[WARN] No instance of {lt} marked `toggleable` in INIT.")
        if ht and not has_cap("pickupable", ht):
            print(f"[WARN] No instance of {ht} marked `pickupable` in INIT.")
    elif task == "pick_two_obj_and_place":
        ot = new_types.get("OBJ_TYPE_1")
        rt = new_types.get("REC_TYPE_1")
        if ot and count_type(ot) < 2:
            print(f"[WARN] Fewer than two instances for {ot} in INIT.")
        if ot and rt and not ok_can(rt, ot):
            print(f"[WARN] canContain({rt}, {ot}) not present in INIT.")

def create_synthetic_task(
    new_types: dict,
    templ: str,
    goal_raw: str,
    task: str,
    S: dict,
    data: dict,
    game_path: str,
    goal_obj_types,
    goal_rec_types,
    dest_split: str,
    project,
    step,
    node_index,
    attempt_id: Optional[int] = None,
):
    if templ and ("{" in templ):
        new_goal_block = fill_goal_from_template(templ, new_types)
    else:
        new_goal_block = replace_types_direct(goal_raw, goal_obj_types, goal_rec_types, new_types)

    if "(:goal" not in new_goal_block:
        new_goal_block = "(:goal\n" + new_goal_block + "\n)"

    new_problem_text = substitute_goal_in_problem(data["pddl_problem"], new_goal_block)

    #new_task_dir, new_trial_dir = synth_task_names_from_path(
    #    game_path, new_types, goal_obj_types, goal_rec_types, dest_split, project, step, node_index
    #)
    new_task_dir, new_trial_dir = synth_task_names_from_path(
        game_path, new_types, goal_obj_types, goal_rec_types, dest_split, project, step, node_index,
        attempt_id=attempt_id,
    )
    os.makedirs(new_trial_dir, exist_ok=True)

    src_trial = os.path.dirname(game_path)
    for fname in os.listdir(src_trial):
        if fname == "game.tw-pddl":
            continue
        src = os.path.join(src_trial, fname)
        dst = os.path.join(new_trial_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    new_data = dict(data)
    new_data["pddl_problem"] = new_problem_text
    new_game_path = os.path.join(new_trial_dir, "game.tw-pddl")
    with open(new_game_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("[INFO] Validating selection against INIT (warnings only)...")
    validate_selection(task, S, new_types)

    print("\n=== DONE ===")
    print("New task folder :", new_task_dir)
    print("New trial folder:", new_trial_dir)
    print("New game file   :", new_game_path)

    return new_task_dir, new_trial_dir, new_game_path, new_data


import re, ast
def parse_two_tokens(s: str):
    if s is None:
        raise ValueError("empty string")
    s = s.strip()
    s = s.replace("，", ",").replace("、", ",")
    s = re.sub(r'^[\[\(\{<\s]+|[\]\)\}>\s]+$', '', s)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            t1 = str(val[0]).strip().strip('\'"`')
            t2 = str(val[1]).strip().strip('\'"`')
            return t1, t2
    except Exception:
        pass
    parts = [p.strip().strip('\'"`') for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Need two tokens separated by a comma, got: {s!r}")
    return parts[0], parts[1]


def purge_bad_trials(base_dir, node_index, step, delete: bool = False):
    import os, json, shutil
    try:
        import textworld
        from textworld import EnvInfos
    except Exception as e:
        raise RuntimeError(f"need to install textworld: {e}")

    suffix = f"-{str(node_index)}-syn-{str(step)}"

    trial_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).endswith(suffix):
            if "game.tw-pddl" in files and "traj_data.json" in files:
                trial_dirs.append(root)
    trial_dirs.sort()

    kept, bad = [], []
    for td in trial_dirs:
        gf = os.path.join(td, "game.tw-pddl")

        ok, reason = True, ""
        try:
            with open(gf, "r") as f:
                data = json.load(f)
            for k in ("pddl_problem", "pddl_domain"):
                if k not in data or not isinstance(data[k], str) or not data[k].strip():
                    ok, reason = False, f"missing/empty '{k}'"
                    break
        except Exception as e:
            ok, reason = False, f"JSONError: {e}"

        if ok:
            try:
                env = textworld.start(gf, EnvInfos())
                env.close()
            except Exception as e:
                ok, reason = False, f"{type(e).__name__}: {e}"

        if ok:
            kept.append(td)
        else:
            bad.append((td, reason))
            print(f"[BAD ] {td} -> {reason}")
            if delete:
                shutil.rmtree(td, ignore_errors=True)
                task_dir = os.path.dirname(td)
                remaining = any(
                    os.path.isdir(os.path.join(task_dir, name)) and
                    os.path.exists(os.path.join(task_dir, name, "game.tw-pddl"))
                    for name in os.listdir(task_dir)
                )
                if not remaining:
                    print(f"[INFO] delet empty task directory: {task_dir}")
                    shutil.rmtree(task_dir, ignore_errors=True)

    print("\n====== results ======")
    print(f"number of trial keep: {len(kept)}")
    print(f"number of trial broken: {len(bad)}")
    return kept, bad


# =========================
# RLVR Co-Evolve: Env State (slot + active + pending + history)
# =========================

def get_env_state_path(env_data_dir: str, project: str, node_index: Optional[int] = None) -> str:
    # per-node state to avoid multi-node race
    if node_index is None:
        return os.path.join(env_data_dir, "json_2.1.1", project, "env_state.json")
    return os.path.join(env_data_dir, "json_2.1.1", project, f"env_state-node{int(node_index)}.json")

def _atomic_json_dump(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}.{int(time.time()*1000)}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_env_state(env_data_dir: str, project: str, node_index: Optional[int] = None) -> Dict[str, Any]:
    path = get_env_state_path(env_data_dir, project, node_index=node_index)
    if not os.path.exists(path):
        return {"version": 2, "meta": {"project": project, "node_index": node_index}, "slots": {}, "pending": []}
    with open(path, "r", encoding="utf-8") as f:
        st = json.load(f)
    if "slots" not in st: st["slots"] = {}
    if "pending" not in st: st["pending"] = []
    if "meta" not in st: st["meta"] = {"project": project, "node_index": node_index}
    if "version" not in st: st["version"] = 2
    return st

def save_env_state(env_data_dir: str, project: str, st: Dict[str, Any], node_index: Optional[int] = None):
    path = get_env_state_path(env_data_dir, project, node_index=node_index)
    _atomic_json_dump(st, path)

def ensure_slot(st: Dict[str, Any], slot_id: str, raw_game_id: str):
    slots = st.setdefault("slots", {})
    if slot_id not in slots:
        slots[slot_id] = {
            "slot_id": slot_id,
            "raw_game_id": raw_game_id,
            "active_game_id": raw_game_id,
            "active_type": "raw",
            "active_epoch": 0,
            "history": [
                {"event": "init", "epoch": 0, "active_game_id": raw_game_id, "active_type": "raw"}
            ],
        }
    else:
        slots[slot_id].setdefault("slot_id", slot_id)
        slots[slot_id].setdefault("raw_game_id", raw_game_id)
        slots[slot_id].setdefault("active_game_id", raw_game_id)
        slots[slot_id].setdefault("active_type", "raw")
        slots[slot_id].setdefault("active_epoch", 0)
        slots[slot_id].setdefault("history", [])

def pending_slot_ids(st: Dict[str, Any]) -> set:
    s = set()
    for p in st.get("pending", []):
        if "slot_id" in p:
            s.add(str(p["slot_id"]))
    return s

def direction_from_acc(acc: float, hi: float = 0.8, lo: float = 0.2):
    """
    Return:
      "harder" if acc >= hi
      "easier" if acc <= lo
      None otherwise (no co-evolve)
    """
    try:
        a = float(acc)
    except Exception:
        return None

    if a >= hi:
        return "harder"
    if a <= lo:
        return "easier"
    return None

def add_pending(st: Dict[str, Any], item: Dict[str, Any]):
    st.setdefault("pending", [])
    sid = str(item.get("slot_id"))
    # one pending per slot: if exists, skip
    for p in st["pending"]:
        if str(p.get("slot_id")) == sid:
            return
    st["pending"].append(item)

def remove_pending_by_game_id(st: Dict[str, Any], game_id: str):
    st["pending"] = [p for p in st.get("pending", []) if p.get("temp_game_id") != game_id and p.get("game_id") != game_id]

def remove_pending_by_slot(st: Dict[str, Any], slot_id: str):
    st["pending"] = [p for p in st.get("pending", []) if str(p.get("slot_id")) != str(slot_id)]

def game_type_from_game_id(game_id: str, project: str) -> str:
    if game_id.startswith(f"{project}/temp_train/"):
        return "temp"
    if game_id.startswith(f"{project}/syn_train/"):
        return "syn"
    if game_id.startswith("train/"):
        return "raw"
    return "unknown"

def move_trial_temp_to_syn(env_data_dir: str, project: str, temp_game_id: str) -> str:
    """
    Move a trial directory from <project>/temp_train/... to <project>/syn_train/... .
    Return the new syn game_id.
    """
    abs_game = resolve_game_id(temp_game_id, env_data_dir)
    if not os.path.exists(abs_game):
        raise FileNotFoundError(abs_game)

    temp_root = os.path.join(env_data_dir, "json_2.1.1", project, "temp_train")
    syn_root  = os.path.join(env_data_dir, "json_2.1.1", project, "syn_train")

    trial_dir = os.path.dirname(abs_game)
    rel_trial = os.path.relpath(trial_dir, temp_root)
    dst_trial_dir = os.path.join(syn_root, rel_trial)

    os.makedirs(os.path.dirname(dst_trial_dir), exist_ok=True)
    if os.path.exists(dst_trial_dir):
        shutil.rmtree(dst_trial_dir, ignore_errors=True)
    shutil.move(trial_dir, dst_trial_dir)

    src_task_dir = os.path.dirname(trial_dir)
    if os.path.isdir(src_task_dir) and not os.listdir(src_task_dir):
        shutil.rmtree(src_task_dir, ignore_errors=True)

    new_game_path = os.path.join(dst_trial_dir, "game.tw-pddl")
    return canonical_game_id(new_game_path, env_data_dir)

def delete_trial_dir_for_game(env_data_dir: str, project: str, game_id: str):
    abs_game = resolve_game_id(game_id, env_data_dir)
    trial_dir = os.path.dirname(abs_game)
    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)
        task_dir = os.path.dirname(trial_dir)
        if os.path.isdir(task_dir) and not os.listdir(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)

__all__ = [
    "parse_facts","base_name","make_aliases",
    "summarize_init_english",
    "detect_task_type_from_path","goal_detect_types","goal_mask_types_consistent",
    "can_contain_ok","has_capability_type","count_instances_of_type",
    "goal_brief_and_instruction",
    "extract_block",
    "_norm_type","decide_new_types","type_to_short","collect_placeholders","fill_goal_from_template",
    "canonical_game_id","resolve_game_id","strip_syn_suffix",
    "replace_types_direct","build_name_with_replacements","synth_task_names_from_path",
    "substitute_goal_in_problem","validate_selection","create_synthetic_task","parse_two_tokens",
    "purge_bad_trials",
    # env state
    "get_env_state_path","load_env_state","save_env_state","ensure_slot",
    "pending_slot_ids","direction_from_acc","add_pending","remove_pending_by_game_id","remove_pending_by_slot",
    "game_type_from_game_id","move_trial_temp_to_syn","delete_trial_dir_for_game",
]
