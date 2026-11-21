#!/usr/bin/env python3
"""Weekly synchronization script for University of Lucknow (lkouniv.ac.in) official documents.

Scrapes official university circulars, examination guidelines, and engineering department
pages, checks for updates via SHA256 checksums, ingests changed documents into Supabase,
and triggers vector index synchronization on the live Hugging Face ZeroGPU backend.
"""

import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Add backend directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Environment configurations
if not os.environ.get("DATABASE_URL"):
    # Attempt to read from .env if present
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

if not os.environ.get("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL environment variable is required to run document sync.")

os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("SECRET_KEY", "REMOVED")
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("COOKIE_SECURE", "true")
os.environ.setdefault("CELERY_EAGER", "True")

BACKEND_URL = os.environ.get("BACKEND_URL", "https://shashank17singh-campusmind.hf.space")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CampusMindBot/1.0"}

TARGET_PAGES = [
    # 1. Central Programmes & Admissions (All University Faculties)
    {
        "url": "https://www.lkouniv.ac.in/en/page/undergraduate-programmes",
        "title": "University of Lucknow - Undergraduate Programmes & Admission Guidelines (BA, BSc, BCom, BTech, LLB, BVA)",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_undergraduate_programmes.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/en/page/postgraduate-programmes",
        "title": "University of Lucknow - Postgraduate Programmes & Admission Guidelines (MA, MSc, MCom, LLM, MEd, MBA, MCA)",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_postgraduate_programmes.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/article/en/course",
        "title": "University of Lucknow - Academic Departments and Course Directory",
        "doc_type": "circular",
        "department": None,
        "filename": "uol_all_departments_courses.txt",
    },
    # 2. Faculty of Arts & Humanities (BA, MA, Languages, Social Sciences)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-arts",
        "title": "University of Lucknow - Faculty of Arts (Humanities, Languages & Social Sciences)",
        "doc_type": "circular",
        "department": "Arts",
        "filename": "uol_faculty_of_arts.txt",
    },
    # 3. Faculty of Science (BSc, MSc, Physics, Chemistry, Math, Life Sciences, Computing)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-science",
        "title": "University of Lucknow - Faculty of Science (Sciences & Computing)",
        "doc_type": "circular",
        "department": "Science",
        "filename": "uol_faculty_of_science.txt",
    },
    # 4. Faculty of Commerce & Business Studies (BCom, MCom, Business)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-commerce",
        "title": "University of Lucknow - Faculty of Commerce (B.Com, M.Com, Business Studies & Finance)",
        "doc_type": "circular",
        "department": "Commerce",
        "filename": "uol_faculty_of_commerce.txt",
    },
    # 5. Faculty of Law (LLB, LLM, 5-Year Integrated Law)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-law",
        "title": "University of Lucknow - Faculty of Law (LL.B., LL.M. & Integrated Law)",
        "doc_type": "circular",
        "department": "Law",
        "filename": "uol_faculty_of_law.txt",
    },
    # 6. Faculty of Education (BEd, MEd)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-education",
        "title": "University of Lucknow - Faculty of Education (B.Ed., M.Ed., Teacher Training)",
        "doc_type": "circular",
        "department": "Education",
        "filename": "uol_faculty_of_education.txt",
    },
    # 7. Faculty of Fine Arts (BVA, MVA, Visual Arts)
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-fine-arts",
        "title": "University of Lucknow - Faculty of Fine Arts (BVA, MVA, Applied & Visual Arts)",
        "doc_type": "circular",
        "department": "Fine Arts",
        "filename": "uol_faculty_of_fine_arts.txt",
    },
    # 8. Faculty of Engineering & Technology (FoET)
    {
        "url": "https://www.lkouniv.ac.in/en/page/engineering-programmes",
        "title": "University of Lucknow - Engineering Programmes & Fee Structure (FoET)",
        "doc_type": "regulation",
        "department": "Engineering",
        "filename": "uol_engineering_programmes_fees.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/en/page/faculty-of-engineering-pages",
        "title": "University of Lucknow - Faculty of Engineering & Technology Overview",
        "doc_type": "circular",
        "department": "Engineering",
        "filename": "uol_foet_overview.txt",
    },
    # 9. Examination & Controller of Examination (University-Wide)
    {
        "url": "https://www.lkouniv.ac.in/en/page/controller-of-examination",
        "title": "University of Lucknow - Examination Guidelines & Controller of Examination",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_examination_guidelines.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/article/en/time-table",
        "title": "University of Lucknow - Examination Time Tables & Schedules (All Faculties)",
        "doc_type": "circular",
        "department": None,
        "filename": "uol_examination_time_tables.txt",
    },
    # 10. Central Campus Facilities, Policies & Student Welfare
    {
        "url": "https://www.lkouniv.ac.in/en/page/tagore-library",
        "title": "University of Lucknow - Tagore Central Library (Timings, Membership & Rules)",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_tagore_library_rules.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/article/en/hostel-details",
        "title": "University of Lucknow - Hostel Rules & Allotment Guidelines",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_hostel_rules_guidelines.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/en/page/placement-cell",
        "title": "University of Lucknow - Central Placement and Career Counselling Cell",
        "doc_type": "notice",
        "department": None,
        "filename": "uol_placement_cell.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/article/en/proctorial",
        "title": "University of Lucknow - Proctorial Board, Student Conduct & Anti-Ragging Guidelines",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_proctorial_anti_ragging.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/article/en/policies",
        "title": "University of Lucknow - Academic Policies and Guidelines",
        "doc_type": "regulation",
        "department": None,
        "filename": "uol_academic_policies.txt",
    },
    {
        "url": "https://www.lkouniv.ac.in/en/page/university-at-a-glance",
        "title": "University of Lucknow - Campus Overview & Administration",
        "doc_type": "notice",
        "department": None,
        "filename": "uol_campus_at_a_glance.txt",
    },
]


def clean_html(html: str) -> str:
    """Extract legible plaintext and structure from HTML markup."""
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<header.*?</header>", "", text, flags=re.S | re.I)
    text = re.sub(r"<nav.*?</nav>", "", text, flags=re.S | re.I)
    text = re.sub(r"<footer.*?</footer>", "", text, flags=re.S | re.I)
    text = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n## \1\n\n", text, flags=re.S | re.I)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.S | re.I)
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join([line for line in lines if line])


def fetch_url(url: str) -> str | None:
    """Fetch URL with timeout and error handling."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}")
        return None


def sync_documents():
    """Main weekly document sync routine."""
    print("=" * 60)
    print(f"CampusMind Weekly UoL Document Sync - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    try:
        from app.db import session_scope
        from app.ingestion.pipeline import run_ingestion
        from app.ingestion.service import create_document
        from app.ingestion.tasks import build_deps
        from app.models import Document
    except ImportError as e:
        print(f"Failed to load backend modules: {e}")
        sys.exit(1)

    deps = build_deps()
    updated_count = 0
    unchanged_count = 0

    with session_scope() as session:
        for target in TARGET_PAGES:
            print(f"\nProcessing: {target['title']} ({target['url']})")
            html = fetch_url(target["url"])
            if not html:
                continue

            cleaned = clean_html(html)
            if len(cleaned) < 150:
                print(f"  [SKIP] Page content too short ({len(cleaned)} chars), possible scrape block.")
                continue

            banner = (
                f"University of Lucknow (UoL) Official Documentation\n"
                f"Source: {target['url']}\n"
            )
            full_content = f"{target['title']}\n\n{banner}\n\n{cleaned}"
            content_bytes = full_content.encode("utf-8")
            content_sha256 = hashlib.sha256(content_bytes).hexdigest()

            # Check if this exact content already exists in Supabase
            existing = session.query(Document).filter(Document.sha256 == content_sha256).first()
            if existing:
                print(f"  [UNCHANGED] Document already up-to-date (ID: {existing.id})")
                unchanged_count += 1
                continue

            # Ingest new or modified version
            print(f"  [UPDATE] Detected new or changed content ({len(full_content)} chars). Ingesting...")
            res = create_document(
                session,
                deps.storage,
                filename=target["filename"],
                data=content_bytes,
                uploader_id=None,
                title=target["title"],
                doc_type=target["doc_type"],
                department=target["department"],
            )
            session.commit()

            if not res.duplicate:
                ok = run_ingestion(deps, res.document.id)
                if ok:
                    print(f"  [SUCCESS] Successfully indexed {target['title']} (Doc ID: {res.document.id})")
                    updated_count += 1
                else:
                    print(f"  [ERROR] Ingestion pipeline failed for {res.document.id}")
            else:
                print(f"  [UNCHANGED] Hash match on storage. Skipped.")
                unchanged_count += 1

    print("\n" + "=" * 60)
    print(f"Sync Complete: {updated_count} updated, {unchanged_count} unchanged.")
    print("=" * 60)

    # Trigger vector index synchronization on live Hugging Face Space backend
    if updated_count > 0:
        sync_url = f"{BACKEND_URL.rstrip('/')}/api/v1/sync"
        print(f"\nTriggering vector store synchronization at: {sync_url}")
        try:
            req = urllib.request.Request(sync_url)
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"Vector sync response: {r.status} {r.read().decode()}")
        except Exception as se:
            print(f"Vector sync trigger notice: {se}")


if __name__ == "__main__":
    sync_documents()
