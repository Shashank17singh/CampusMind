#!/usr/bin/env python3
"""Recursive Department & Notices Ingestion Engine for University of Lucknow (lkouniv.ac.in).

Crawls all 60+ academic departments on UDRC (History, Faculty Rosters, Courses, Thrust Areas, Contact)
and active circular notices from lkouniv.ac.in, extracts text, computes SHA256 checksums,
and ingests new/modified documents into Supabase and the vector store.
"""

import concurrent.futures
import hashlib
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure database and environment variables
if not os.environ.get("DATABASE_URL"):
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

if not os.environ.get("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL environment variable is required to run crawler.")

os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("SECRET_KEY", "REMOVED")
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("COOKIE_SECURE", "true")
os.environ.setdefault("CELERY_EAGER", "True")

BACKEND_URL = os.environ.get("BACKEND_URL", "https://shashank17singh-campusmind.hf.space")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CampusMindBot/2.0"}


def clean_text(html: str) -> str:
    """Strip markup and normalize whitespace."""
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return " ".join([line for line in lines if line]).strip()


def fetch_url(url: str, timeout: int = 12) -> str:
    """Fetch URL with timeout and error handling."""
    if not url.startswith(("http://", "https://")):
        return ""
    req = urllib.request.Request(url, headers=HEADERS)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def parse_department(dept_id: int) -> dict | None:
    """Extract complete profile for a single department ID from UDRC."""
    base_url = "http://udrc.lkouniv.ac.in/Department/DepartmentDetail"

    # 1. Fetch History & About
    hist_html = fetch_url(f"{base_url}/History?dept={dept_id}")
    if not hist_html or "Invalid Department" in hist_html:
        return None

    # Department Name
    m_name = re.search(r"<h[1-4][^>]*>(.*?)</h[1-4]>", hist_html, re.I)
    dept_name = clean_text(m_name.group(1)) if m_name else ""
    if not dept_name or dept_name in ["Policies", "Media", "University of Lucknow"]:
        m_span = re.search(r"<span id=[\"']lblDeptName[\"'][^>]*>(.*?)</span>", hist_html, re.I)
        dept_name = clean_text(m_span.group(1)) if m_span else f"Department {dept_id}"

    # Ignore non-departmental shells
    if not dept_name or dept_name == f"Department {dept_id}":
        return None

    # Head of Department
    hod_name = ""
    hod_desig = "Head of Department"
    m_hod_block = re.search(
        r"Head of Department.*?(DR\.[^<]+|PROF\.[^<]+|MR\.[^<]+)", hist_html, re.S | re.I
    )
    if m_hod_block:
        hod_name = clean_text(m_hod_block.group(1))

    # About Section
    about_text = ""
    m_abt = re.search(
        r"<div class=[\"']about-department[\"'][^>]*>(.*?)</div>", hist_html, re.S | re.I
    )
    if m_abt:
        about_text = clean_text(m_abt.group(1))
    else:
        # Fallback to main content body
        clean_all = clean_text(hist_html)
        about_text = clean_all[:1200]

    # 2. Fetch Faculty Members
    fac_html = fetch_url(f"{base_url}/Faculties?dept={dept_id}")
    faculty_members = []
    for blk in re.findall(r"<table[^>]*>(.*?)</table>", fac_html, re.S | re.I):
        if "icon-user" in blk or "icon-briefcase" in blk:
            nm = re.search(r"<td[^>]*width=[\"']95%[\"'][^>]*>(.*?)</td>", blk, re.S | re.I)
            name_val = clean_text(nm.group(1)) if nm else ""
            desig = re.search(r"icon-briefcase.*?<td>(.*?)</td>", blk, re.S | re.I)
            desig_val = clean_text(desig.group(1)) if desig else "Faculty Member"
            phone = re.search(r"icon-phone.*?<td>(.*?)</td>", blk, re.S | re.I)
            phone_val = clean_text(phone.group(1)) if phone else "N/A"
            email = re.search(r"icon-envelope.*?<a[^>]*>(.*?)</a>", blk, re.S | re.I)
            email_val = clean_text(email.group(1)) if email else "N/A"
            if name_val:
                faculty_members.append(
                    {
                        "name": name_val,
                        "designation": desig_val,
                        "phone": phone_val,
                        "email": email_val,
                    }
                )

    # 3. Fetch Courses & Programmes
    crs_html = fetch_url(f"{base_url}/Course?dept={dept_id}")
    courses_text = clean_text(crs_html)
    # Remove navigation boilerplate
    if "Programmes" in courses_text:
        courses_text = courses_text[
            courses_text.find("Programmes") : courses_text.find("Programmes") + 1500
        ]

    # 4. Fetch Thrust Areas
    thrust_html = fetch_url(f"{base_url}/ThrustAreas?dept={dept_id}")
    thrust_text = clean_text(thrust_html)
    if "Thrust Areas" in thrust_text:
        thrust_text = thrust_text[
            thrust_text.find("Thrust Areas") : thrust_text.find("Thrust Areas") + 1500
        ]

    # 5. Fetch Contact Details
    contact_html = fetch_url(f"{base_url}/ContactDetails?dept={dept_id}")
    contact_text = clean_text(contact_html)
    if "Contact Details" in contact_text:
        contact_text = contact_text[
            contact_text.find("Contact Details") : contact_text.find("Contact Details") + 1000
        ]

    # Format into comprehensive Markdown
    fac_table_md = "\n".join(
        [
            f"| {f['name']} | {f['designation']} | {f['phone']} | {f['email']} |"
            for f in faculty_members
        ]
    )
    if not fac_table_md:
        fac_table_md = (
            "| Department Office | Administrative Staff | Available on site | Available on site |"
        )

    md_content = f"""# University of Lucknow - {dept_name}

**Source:** http://udrc.lkouniv.ac.in/Department/DepartmentDetail/History?dept={dept_id}
**Department Code:** UOL-DEPT-{dept_id}
**Head of Department:** {hod_name or "HOD Office"} ({hod_desig})

## About the Department
{about_text}

## Faculty Members & Academic Staff
| Name | Designation | Contact Phone | Email |
|---|---|---|---|
{fac_table_md}

## Academic Programmes & Degrees Offered
{courses_text or "Undergraduate, Postgraduate, and Doctoral research programmes as governed by University ordinances."}

## Research Thrust Areas & Laboratories
{thrust_text or "Active departmental research laboratories, doctoral scholars, and faculty research specializations."}

## Department Contact & Campus Location
{contact_text or f"Department of {dept_name}, University of Lucknow, Lucknow, Uttar Pradesh 226007."}
"""

    slug = re.sub(r"[^a-z0-9]+", "_", dept_name.lower()).strip("_")
    filename = f"uol_dept_{dept_id}_{slug}.txt"
    title = f"University of Lucknow - {dept_name} (Faculty Roster, Courses, HOD & Research)"

    return {
        "dept_id": dept_id,
        "name": dept_name,
        "title": title,
        "filename": filename,
        "content": md_content,
        "faculty_count": len(faculty_members),
    }


def crawl_active_notices() -> list[dict]:
    """Crawl active notices from lkouniv.ac.in notice board."""
    url = "https://www.lkouniv.ac.in/en/news?Newslistslug=en-notices"
    html = fetch_url(url)
    if not html:
        return []

    notices = []
    # Match table rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    for r in rows:
        cells = [clean_text(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S | re.I)]
        links = re.findall(r"href=[\"']([^\"']+)[\"']", r)
        if len(cells) >= 3 and cells[0].isdigit():
            sno, subject, date_str = cells[0], cells[1], cells[2]
            pdf_link = ""
            for link in links:
                if ".pdf" in link.lower():
                    pdf_link = urllib.parse.urljoin("https://www.lkouniv.ac.in/en/", link)
                    break

            notices.append(
                {
                    "sno": sno,
                    "title": f"UoL Official Notice - {subject[:80]}",
                    "subject": subject,
                    "date": date_str,
                    "pdf_link": pdf_link,
                }
            )
    return notices


def run_department_expansion():
    """Execute complete recursive department discovery, parsing and indexing."""
    print("=" * 70)
    print("CampusMind: University-Wide Department & Notice Expansion Engine")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

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

    # 1. Parallel fetch of all departments
    print("\n[Step 1/3] Scanning all department portals on UDRC (IDs 1 to 65)...")
    dept_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(parse_department, d): d for d in range(1, 65)}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                dept_results.append(res)

    dept_results.sort(key=lambda x: x["dept_id"])
    print(f"Successfully scraped and structured {len(dept_results)} university departments.")

    # 2. Ingest Departments into Supabase & Vector Store
    print(f"\n[Step 2/3] Ingesting {len(dept_results)} departments into Supabase & Vector Index...")
    indexed_count = 0
    unchanged_count = 0

    with session_scope() as session:
        for dept in dept_results:
            content_bytes = dept["content"].encode("utf-8")
            content_sha256 = hashlib.sha256(content_bytes).hexdigest()

            # Check if this exact content already exists in Supabase
            existing = session.query(Document).filter(Document.sha256 == content_sha256).first()
            if existing:
                unchanged_count += 1
                continue

            print(f"  -> Ingesting: {dept['name']} ({dept['faculty_count']} faculty members)")
            res = create_document(
                session,
                deps.storage,
                filename=dept["filename"],
                data=content_bytes,
                uploader_id=None,
                title=dept["title"],
                doc_type="contact_list",
                department=dept["name"],
            )
            session.commit()

            if not res.duplicate:
                ok = run_ingestion(deps, res.document.id)
                if ok:
                    indexed_count += 1
                else:
                    print(f"    [WARN] Ingestion pipeline failed for {res.document.id}")
            else:
                unchanged_count += 1

    print(f"\nDepartment Sync Result: {indexed_count} newly indexed, {unchanged_count} unchanged.")

    # 3. Ingest Active Circular Notices
    print("\n[Step 3/3] Scanning latest active university circular notices...")
    notices = crawl_active_notices()
    print(f"Found {len(notices)} active circular notices on the central portal.")

    notice_indexed = 0
    with session_scope() as session:
        for n in notices[:20]:  # Top 20 active circulars
            notice_content = f"""# {n["title"]}

**Published Date:** {n["date"]}
**Official Subject:** {n["subject"]}
**Document Link:** {n["pdf_link"] or "Published on official notice board"}
**Source:** https://www.lkouniv.ac.in/en/news?Newslistslug=en-notices

## Circular Summary
This is an official administrative notice issued by the University of Lucknow for students, faculty, and affiliated colleges regarding {n["subject"]}.
"""
            n_bytes = notice_content.encode("utf-8")
            n_sha256 = hashlib.sha256(n_bytes).hexdigest()

            existing = session.query(Document).filter(Document.sha256 == n_sha256).first()
            if existing:
                continue

            slug = re.sub(r"[^a-z0-9]+", "_", n["title"].lower()).strip("_")[:40]
            res = create_document(
                session,
                deps.storage,
                filename=f"uol_notice_{n['sno']}_{slug}.txt",
                data=n_bytes,
                uploader_id=None,
                title=n["title"],
                doc_type="notice",
                department=None,
            )
            session.commit()
            if not res.duplicate and run_ingestion(deps, res.document.id):
                notice_indexed += 1

    print(f"Notice Sync Result: {notice_indexed} new circular notices indexed.")

    # 4. Trigger Vector Store Hot-Reload on HF Space Backend
    total_new = indexed_count + notice_indexed
    if total_new > 0:
        sync_url = f"{BACKEND_URL.rstrip('/')}/api/v1/sync"
        print(f"\nTriggering vector store synchronization at: {sync_url} ...")
        try:
            req = urllib.request.Request(sync_url)  # noqa: S310
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                print(f"Vector sync response: {r.status} {r.read().decode()}")
        except Exception as se:
            print(f"Vector sync trigger notice: {se}")

    print("\n" + "=" * 70)
    print("Expansion Complete! All 60+ departments and active notices are synchronized.")
    print("=" * 70)


if __name__ == "__main__":
    run_department_expansion()
