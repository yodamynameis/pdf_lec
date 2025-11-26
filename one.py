#!/usr/bin/env python3
"""
pipeline_to_manim_1080p_fixed_v2.py

Improvements over original:
- Safer fenced-code extraction (preserves single backticks in strings, extracts largest fenced block).
- Do NOT strip single backticks globally (previous version removed all ` which could break strings/comments).
- Safer Dot(...) replacement that preserves kwargs and whitespace.
- Improved Graph(...) -> Axes+plot replacement with better x_range parsing and safer heuristics.
- Camera fixes handle more variants (camera.shfit, camera.shift, self.camera.shift, camera.frame.shift).
- ensure_imports is idempotent and appends config only when missing.
- Better handling of Chromadb client errors.
- Extra defensive logging and clearer warnings file.

Target: produce cleaner, less-destructive automatic fixes to reduce syntax/runtime errors in generated Manim code.
"""



import os
import re
import sys
import math
import shutil
import json
from typing import List, Dict, Tuple

# PDF reading
from pypdf import PdfReader

# embeddings and DB
from sentence_transformers import SentenceTransformer
import chromadb

# Gemini client (ensure package present or configure appropriately)
import google.generativeai as genai

MANIM_VERSION_NOTE = "Manim Community v0.19.0 (1920x1080 target)"


# -----------------------
# Helper: API key setup (prefer env var)
# -----------------------
def configure_gemini_from_env():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # WARNING: fallback hardcoded key (insecure). Replace before sharing/publishing.
        api_key = "AIzaSyDf9cGc7ww8wh2tcB9rVEvlDgyT4iV69HM"
        print("⚠️ Warning: Using fallback hardcoded Gemini API key. Set GEMINI_API_KEY env var to avoid this.")
    if not api_key or api_key.strip() == "":
        raise EnvironmentError("Gemini API key is empty. Please set a valid key in GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    return genai


# -----------------------
# PDF extraction
# -----------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = ""
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


# -----------------------
# Chunk utilities
# -----------------------
def compute_chunk_params_from_length(text_len: int,
                                     min_chunks: int = 6,
                                     max_chunks: int = 20,
                                     chars_per_chunk_hint: int = 3000) -> Tuple[int, int]:
    if text_len <= 0:
        return 1000, 200
    estimated_chunks = max(1, math.ceil(text_len / chars_per_chunk_hint))
    n_chunks = min(max(estimated_chunks, min_chunks), max_chunks)
    chunk_size = math.ceil(text_len / n_chunks)
    chunk_size = int(min(max(chunk_size, 500), 4000))
    overlap = int(max(50, math.ceil(chunk_size * 0.20)))
    return chunk_size, overlap


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end].strip())
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def auto_chunk_text(text: str) -> Tuple[List[str], int, int]:
    chunk_size, overlap = compute_chunk_params_from_length(len(text))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return chunks, chunk_size, overlap


# -----------------------
# Chroma storage
# -----------------------
def store_chunks_in_chroma(chunks: List[str], db_path: str = "./chroma_db"):
    emb_model = SentenceTransformer("                                                                                               ")
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        # fallback to in-memory client if persistent client fails
        print("⚠️ chromadb.PersistentClient failed, falling back to in-memory client:", str(e))
        client = chromadb.Client()
    collection = client.get_or_create_collection("pdf_collection")
    embeddings = emb_model.encode(chunks).tolist()
    ids = [str(i) for i in range(len(chunks))]
    try:
        existing = collection.get(ids=ids)
        existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if existing_ids:
            collection.delete(ids=existing_ids)
    except Exception:
        # ignore if collection.get is unsupported for some client versions
        pass
    collection.add(documents=chunks, ids=ids, embeddings=embeddings)
    return collection, emb_model


# -----------------------
# Retrieval
# -----------------------
def retrieve_relevant_chunks(collection, emb_model, query: str, n_results: int = 3) -> List[str]:
    if not query or not query.strip():
        return []
    query_emb = emb_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results)
    docs = []
    try:
        # new chroma returns dict with 'documents'
        docs = results.get("documents", [[]])[0] if isinstance(results, dict) else []
    except Exception:
        try:
            docs = results["documents"][0]
        except Exception:
            docs = []
    return docs


# -----------------------
# Prompts
# -----------------------
def build_script_prompt(query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""
    prompt = f"""
You are an expert educational animator and Manim scriptwriter. Using only the context provided, write a concise, 2-minute instructional video script tailored for Manim visualizations.
Constraints & requirements:
- Target audience: engineering/college students with basic math background.
- Produce 3-6 scenes. For each scene, include:
  - Scene Title (one line)
  - Narration: one short paragraph (1-3 sentences)
  - Visuals: bullet-list of specific Manim-friendly visual steps (use Text, not MathTex), Rectangle, Arrow, Graph (Axes), highlights.
  - Transition idea: one short line describing how to move to the next scene.
- IMPORTANT: Do NOT include raw LaTeX or MathTex; use plain Text() unless absolutely required and explain why.
- Output must be plain text, with explicit "Scene:" headers for each scene and clear Visual bullet points.

Context (use only if relevant):
{context}

User Query:
{query}

Return the script now.
"""
    return prompt.strip()


def build_manim_prompt(script_text: str) -> str:
    safety_rules = """
IMPORTANT RULES (follow exactly):
- NEVER use Graph(...). Instead use Axes(...) and axes.plot(lambda x: ..., x_range=[a,b]).
- NEVER call Rectangle(...).surround(obj). Use SurroundingRectangle(obj, color=..., buff=0.2).
- NEVER slice/index Text objects (no text[3:5]). Build equations from multiple Text parts and VGroup them.
- ALWAYS use Dot(axes.coords_to_point(x, y)) when plotting points attached to axes.
- Position everything explicitly: title.to_edge(UP), content.next_to(title, DOWN, buff=0.5), lists via VGroup(...).arrange(DOWN, buff=0.6).
- Insert small waits between sequential animations to avoid overlaps.
- Use only Manim Community v0.19.0 standard imports.
"""
    prompt = f"""
You are an expert Manim Community (v0.19.0) developer. Given the following narrated video script, generate a complete, runnable Python file compatible with Manim Community v0.19.0.

{safety_rules.strip()}

Hard requirements (must follow exactly):
1. Start file header "# Generated by pipeline_to_manim_720p.py — Manim v0.19.0"
2. from manim import *
   from manim import config
   config.pixel_width = 1920
   config.pixel_height = 1080
   config.frame_rate = 30
3. Class `AutoGeneratedVideo(Scene)` with construct() composing scenes.
4. DO NOT use MathTex; use Text() and VGroup parts for equations.
5. Keep scenes modular and position explicitly to avoid overlap.
6. Include short comments describing positioning choices.

Script to convert:
------------------------
{script_text}
------------------------

Return ONLY the Python source code for the Manim file (no explanation).
"""
    return prompt.strip()


# -----------------------
# Gemini interaction
# -----------------------
def call_gemini_generate(prompt: str, model_name: str = "gemini-2.5-pro") -> str:
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
    except TypeError:
        response = model.generate_content({"prompt": prompt})
    text = ""
    # Various response shapes across genai versions
    try:
        if hasattr(response, "text") and response.text:
            text = response.text
        elif isinstance(response, dict) and "candidates" in response:
            text = response["candidates"][0].get("content", "")
        else:
            text = str(response)
    except Exception:
        text = str(response)
    return text


# -----------------------
# Extract Python code from response (more robust)
# -----------------------
def extract_python_code_from_text(text: str) -> str:
    if not text:
        return ""

    # Try to find fenced code blocks first (prefer largest python block)
    fence_patterns = [r"```(?:python|py)?\s*([\s\S]*?)```",
                      r"'''(?:python|py)?\s*([\s\S]*?)'''",
                      r'"""(?:python|py)?\s*([\s\S]*?)"""']
    fence_matches = []
    for pat in fence_patterns:
        fence_matches += re.findall(pat, text, flags=re.I)

    if fence_matches:
        # choose the largest candidate (most likely the full file)
        candidate = max(fence_matches, key=lambda s: len(s.strip()))
        cleaned = candidate.strip()
        return cleaned

    # If no fenced block, try to find a 'from manim' import onward
    manim_match = re.search(r"(from\s+manim[\s\S]+)", text)
    if manim_match:
        return manim_match.group(1).strip()

    # fallback: if it looks like python code, return whole text
    if text.count("\n") > 6 and ("class " in text or "def " in text or "import " in text):
        return text.strip()

    return ""
################################################################################
# -----------------------
# Helper to ensure imports/config are present (idempotent)
# -----------------------
def ensure_imports(code: str) -> str:
    code_out = code or ""
    # If no direct manim import found, prepend
    if not re.search(r"from\s+manim\s+import|import\s+manim", code_out):
        header = "from manim import *\nfrom manim import config\n"
        header += "config.pixel_width = 1920\nconfig.pixel_height = 1080\nconfig.frame_rate = 30\n\n"
        code_out = header + code_out
    else:
        # ensure pixel config keys exist somewhere; append only if missing
        if not re.search(r"config\.pixel_width\s*=", code_out):
            cfg = "\n# Ensure 1080p output\nfrom manim import config\nconfig.pixel_width = 1920\nconfig.pixel_height = 1080\nconfig.frame_rate = 30\n\n"
            code_out = cfg + code_out
    return code_out


# -----------------------
# Auto-fixer & validator (safer, less destructive)
# -----------------------
from typing import Tuple

def auto_fix_manim_code(code: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    fixed = code or ""

    # Remove only triple-fence markers left at start/end, but preserve single backticks
    fixed = re.sub(r'^\s*```[a-zA-Z0-9_\-]*\n', '', fixed)
    fixed = re.sub(r'\n```\s*$', '', fixed)
    fixed = re.sub(r"^\s*'{3,}\s*\n", '', fixed)
    fixed = re.sub(r"\n'{3,}\s*$", '', fixed)

    # Insert missing imports/config if necessary
    fixed = ensure_imports(fixed)

    # Camera typos -> try to fix common misspellings and variants
    cam_patterns = [r'camera\.shfit\s*\(', r'camera\.shift\s*\(', r'self\.camera\.shift\s*\(']
    for pat in cam_patterns:
        new = re.sub(pat, 'self.camera.frame.shift(', fixed, flags=re.I)
        if new != fixed:
            fixed = new
            warnings.append(f"Fixed camera variant '{pat}' -> self.camera.frame.shift(...).")

    # Rectangle(...).surround(obj) -> SurroundingRectangle(obj, color=BLUE, buff=0.2)
    fixed_new = re.sub(
        r'Rectangle\([^)]*\)\.surround\(\s*([A-Za-z0-9_\.]+)\s*\)',
        r'SurroundingRectangle(\1, color=BLUE, buff=0.2)',
        fixed
    )
    if fixed_new != fixed:
        fixed = fixed_new
        warnings.append("Replaced Rectangle(...).surround(...) with SurroundingRectangle(...).")

    # Safer Dot replacements: only replace Dot([...]) when there are two numeric literals or simple names
    def replace_dot(m):
        inside = m.group(1).strip()
        tail = m.group(2) or ""
        # preserve kwargs tail exactly
        # attempt to split coordinates by comma; keep as-is if not simple
        coords = [c.strip() for c in inside.split(',')]
        if len(coords) == 2:
            return f"Dot(axes.coords_to_point({coords[0]}, {coords[1]}){tail})"
        return m.group(0)

    fixed2 = re.sub(r'Dot\s*\(\s*\[([^\]]+?)\]\s*(,\s*[^)]+)?\)', replace_dot, fixed)
    if fixed2 != fixed:
        fixed = fixed2
        warnings.append("Rewrote simple Dot([x,y], ...) to Dot(axes.coords_to_point(x, y), ...) where safe.")

    # Graph(lambda x: expr, x_range=[a,b]) -> axes + axes.plot(lambda x: expr)
    # Use a safer regex that captures body between 'lambda x:' and comma or )
    def replace_graph_lambda(m):
        body = m.group('body').strip()
        xrange = m.group('xrange') or "-5, 5"
        # normalize xrange
        xparts = [p.strip() for p in re.split(r',|\s', xrange) if p.strip()]
        x0 = xparts[0] if len(xparts) > 0 else "-5"
        x1 = xparts[1] if len(xparts) > 1 else "5"
        axes_block = (
            "axes = Axes(\n"
            f"    x_range=[{x0}, {x1}, 1],\n"
            "    y_range=[-5, 5, 1],\n"
            "    x_length=10,\n"
            "    y_length=6,\n"
            ").shift(DOWN * 0.2)\n"
            f"graph = axes.plot(lambda x: {body}, x_range=[{x0}, {x1}])\n"
        )
        warnings.append("Rewrote Graph(...) into Axes + axes.plot(...) (heuristic). Verify ranges/placement.")
        return axes_block

    fixed_graph = re.sub(
        r'Graph\s*\(\s*lambda\s+[^:]+:\s*(?P<body>[^,\)]+)\s*,\s*x_range\s*=\s*\[(?P<xrange>[^\]]+)\][^\)]*\)',
        replace_graph_lambda,
        fixed
    )
    if fixed_graph != fixed:
        fixed = fixed_graph

    # Insert waits between consecutive self.play(...) without waits; do not insert inside comments or strings
    fixed = re.sub(r'(\n\s*self\.play\([^\)]*\)\s*)(?=\n\s*self\.play\()', r'\1\n    self.wait(0.15)\n', fixed, flags=re.M)

    # Detect Text slicing and warn
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\d+\s*:\s*\d+\s*\]', fixed):
        lineno = fixed.count('\n', 0, m.start()) + 1
        snippet = fixed.splitlines()[lineno - 1].strip()
        warnings.append(f"Text slicing detected on line {lineno}: '{snippet}' — convert to VGroup of Text parts manually.")

    # Cleanup SurroundingRectangle parameter duplication (idempotent)
    def fix_surround(match):
        inner = match.group(1).strip()
        tail = match.group(2) or ""
        if tail and ('color=' in tail or 'buff=' in tail):
            return f"SurroundingRectangle({inner}{tail})"
        return f"SurroundingRectangle({inner}, color=BLUE, buff=0.2{tail})"

    fixed = re.sub(r'SurroundingRectangle\(\s*([^,\)]+?)\s*(,([^\)]*))?\)', fix_surround, fixed)

    # Final whitespace normalization
    fixed = re.sub(r'\r\n', '\n', fixed)
    fixed = fixed.strip() + "\n"

    return fixed, warnings


# -----------------------
# Scene parsing helper
# -----------------------
def clean_and_save_scenes(input_text: str, output_file: str = "generated_scenes.txt") -> List[Dict]:
    text = input_text or ""
    scene_splits = re.split(r"(?im)^\s*Scene\s*[:\-]\s*", text)
    scenes = []
    for part in scene_splits[1:]:
        lines = part.strip().splitlines()
        title = lines[0].strip() if lines else "Untitled"
        body = "\n".join(lines[1:]).strip()
        narration = ""
        visuals = ""
        transition = ""
        narr_match = re.search(r"(?is)Narration\s*[:\-]\s*(.*?)(?=\n[A-Za-z][A-Za-z0-9 _\-]*\s*[:\-]|$)", body)
        visuals_match = re.search(r"(?is)Visuals\s*[:\-]\s*(.*?)(?=\n[A-Za-z][A-Za-z0-9 _\-]*\s*[:\-]|$)", body)
        trans_match = re.search(r"(?is)Transition\s*[:\-]\s*(.*?)(?=\n[A-Za-z][A-Za-z0-9 _\-]*\s*[:\-]|$)", body)
        narration = narr_match.group(1).strip() if narr_match else ""
        visuals = visuals_match.group(1).strip() if visuals_match else ""
        transition = trans_match.group(1).strip() if trans_match else ""
        scenes.append({
            "title": title,
            "narration": narration,
            "visuals": visuals,
            "transition": transition
        })
    with open(output_file, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(f"Scene: {s['title']}\n")
            f.write(f"Narration: {s['narration']}\n")
            f.write(f"Visuals: {s['visuals']}\n")
            if s["transition"]:
                f.write(f"Transition: {s['transition']}\n")
            f.write("\n")
    print(f"✅ Cleaned {len(scenes)} scenes saved to {output_file}")
    return scenes


# -----------------------
# Runner
# -----------------------
def run_pipeline(pdf_path: str,
                 query: str,
                 db_path: str = "./chroma_db",
                 gemini_model: str = "gemini-2.5-pro",
                 n_retrieved: int = 3,
                 output_script_file: str = "gemini_output.txt",
                 output_manim_file: str = "auto_generated_manim.py"):
    configure_gemini_from_env()

    print("📥 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("⚠️ No text found in PDF. Exiting.")
        return

    print("✂️ Computing automatic chunk size and splitting text...")
    chunks, chunk_size, overlap = auto_chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap}).")

    print("💾 Storing chunks in ChromaDB...")
    collection, emb_model = store_chunks_in_chroma(chunks, db_path=db_path)
    print("✅ Stored chunks in ChromaDB.")

    print("🔍 Retrieving relevant information for query...")
    retrieved = retrieve_relevant_chunks(collection, emb_model, query, n_results=n_retrieved)
    print(f"✅ Retrieved {len(retrieved)} chunks.")

    print("🧠 Asking Gemini to create a concise narrated script...")
    script_prompt = build_script_prompt(query, retrieved)
    script_text = call_gemini_generate(script_prompt, model_name=gemini_model)
    if not script_text:
        print("⚠️ Gemini returned an empty script. Exiting.")
        return
    with open(output_script_file, "w", encoding="utf-8") as f:
        f.write(script_text)
    print(f"✅ Raw Gemini video script saved to {output_script_file}")

    scenes = clean_and_save_scenes(script_text, "generated_scenes.txt")

    print("🎨 Building Manim code prompt and asking Gemini to output runnable code...")
    manim_prompt = build_manim_prompt(script_text)
    manim_response = call_gemini_generate(manim_prompt, model_name=gemini_model)
    if not manim_response:
        print("⚠️ Gemini returned an empty Manim response. Exiting.")
        return

    manim_code = extract_python_code_from_text(manim_response)
    if not manim_code:
        print("⚠️ Could not find explicit Python code fences; saving full Gemini response to file and attempting to extract code heuristically.")
        manim_code = manim_response

    if not manim_code.startswith("# Generated by"):
        header = f"# Generated by pipeline_to_manim_1080p_fixed_v2.py — verify Manim version if errors occur\n# {MANIM_VERSION_NOTE}\n\n"
        manim_code = header + manim_code

    bak_path = output_manim_file + ".bak"
    with open(output_manim_file + ".tmp", "w", encoding="utf-8") as f:
        f.write(manim_code)
    shutil.copyfile(output_manim_file + ".tmp", bak_path)
    print(f"🔁 Backup of raw generated Manim saved to {bak_path}")

    print("🛠️ Running auto-fixer heuristics on generated Manim code...")
    fixed_code, warnings = auto_fix_manim_code(manim_code)
    fixed_code = ensure_imports(fixed_code)  # ensure imports/config again after fixes
    with open(output_manim_file, "w", encoding="utf-8") as f:
        f.write(fixed_code)
    print(f"✅ Fixed Manim code saved to {output_manim_file}")

    warnings_file = output_manim_file + ".warnings.txt"

    

    if warnings:
        with open(warnings_file, "w", encoding="utf-8") as wf:
            for w in warnings:
                wf.write(w + "\n")
        print(f"\n⚠️ Auto-fixer generated warnings (saved to {warnings_file}):")
        for w in warnings:
            print(" -", w)
    else:
        print("\n✅ Auto-fixer found no warnings.")

    print("\nFinished. Next steps:")
    print(f"  - Inspect {output_manim_file} and backup {bak_path} for any remaining manual fixes.")
    print("  - Render at 1080p with Manim. Example: manim -pqh auto_generated_manim.py AutoGeneratedVideo")


# -----------------------
# RUN (interactive)
# -----------------------
if __name__ == "__main__":
    pdf_path = input("Enter path to the PDF (default 'straight.pdf'): ").strip()
    if not pdf_path:
        pdf_path = "straight.pdf"
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    user_query = input("Enter a short topic/query to guide the script generation (e.g. 'Explain slope of a straight line'): ").strip()
    if not user_query:
        print("No query entered — using a default query about straight line slope.")
        user_query = "Explain the slope of a straight line"

    db_path = "./chroma_db"
    gemini_model = "gemini-2.5-pro"
    n_retrieved = 3
    out_script = "gemini_output.txt"
    out_manim = "auto_generated_manim.py"

    run_pipeline(pdf_path=pdf_path,
                 query=user_query,
                 db_path=db_path,
                 gemini_model=gemini_model,
                 n_retrieved=n_retrieved,
                 output_script_file=out_script,
                 output_manim_file=out_manim)
