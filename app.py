import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel

load_dotenv()

# ────────────────────────────── Config ──────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-please-change")
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("ここに"):
    raise RuntimeError("環境変数 GEMINI_API_KEY を設定してください")

PROJECT_DIR = Path(__file__).parent
YT_DLP = Path(sys.executable).parent / "yt-dlp"


def find_ffmpeg() -> Path:
    env = os.getenv("FFMPEG_PATH")
    if env and Path(env).exists():
        return Path(env)
    local = PROJECT_DIR / "bin" / "ffmpeg"
    if local.exists():
        return local
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return Path(sys_ffmpeg)
    return PROJECT_DIR / "bin" / "ffmpeg"


FFMPEG = find_ffmpeg()
FFMPEG_DIR = FFMPEG.parent

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"
MAX_INLINE_BYTES = 18 * 1024 * 1024
INSIGHT_TRIGGER_DELTA = 3
INSIGHT_TOP_N = 20
METRICS_REFRESH_INTERVAL = 60 * 60
METRICS_REFRESH_BATCH = 3
METRICS_AGE_THRESHOLD_SEC = 6 * 60 * 60


# ────────────────────────────── DB Layer ──────────────────────────────

USE_PG = DATABASE_URL.startswith("postgres")
if USE_PG:
    import psycopg
    from psycopg.rows import dict_row

    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
    PH = "%s"
else:
    DB_PATH = PROJECT_DIR / "knowledge.db"
    PH = "?"


def get_conn():
    if USE_PG:
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def row_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def safe_add_columns(cursor, table: str, cols: dict):
    for col, definition in cols.items():
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        except Exception as e:
            msg = str(e).lower()
            if "already exists" not in msg and "duplicate column" not in msg:
                raise


def init_db() -> None:
    conn = get_conn()
    cursor = conn.cursor()

    if USE_PG:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id BIGSERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT,
                thumbnail TEXT,
                uploader TEXT,
                duration INTEGER,
                view_count BIGINT DEFAULT 0,
                like_count BIGINT DEFAULT 0,
                comment_count BIGINT DEFAULT 0,
                is_own INTEGER DEFAULT 0,
                original TEXT NOT NULL,
                variants TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_by TEXT,
                created_at TEXT NOT NULL,
                metrics_updated_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS insights (
                id BIGSERIAL PRIMARY KEY,
                report TEXT NOT NULL,
                summary TEXT NOT NULL,
                entries_analyzed INTEGER NOT NULL,
                total_views BIGINT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    else:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT,
                thumbnail TEXT,
                uploader TEXT,
                duration INTEGER,
                view_count INTEGER DEFAULT 0,
                like_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                is_own INTEGER DEFAULT 0,
                original TEXT NOT NULL,
                variants TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_by TEXT,
                created_at TEXT NOT NULL,
                metrics_updated_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report TEXT NOT NULL,
                summary TEXT NOT NULL,
                entries_analyzed INTEGER NOT NULL,
                total_views INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    big = "BIGINT" if USE_PG else "INTEGER"
    safe_add_columns(cursor, "entries", {
        "view_count": f"{big} DEFAULT 0",
        "like_count": f"{big} DEFAULT 0",
        "comment_count": f"{big} DEFAULT 0",
        "is_own": "INTEGER DEFAULT 0",
        "metrics_updated_at": "TEXT",
        "created_by": "TEXT",
    })

    conn.commit()
    conn.close()


init_db()
app = FastAPI(title="バズ台本AI")


# ────────────────────────────── Auth ──────────────────────────────

serializer = URLSafeSerializer(SESSION_SECRET, salt="session-v1")


def make_token(username: str) -> str:
    return serializer.dumps({"u": username})


def parse_token(token: str) -> Optional[str]:
    try:
        return serializer.loads(token).get("u")
    except (BadSignature, Exception):
        return None


def current_user(session: Optional[str] = Cookie(None)) -> str:
    if not session:
        raise HTTPException(status_code=401, detail="ログインが必要です")
    user = parse_token(session)
    if not user:
        raise HTTPException(status_code=401, detail="セッションが無効です")
    return user


# ────────────────────────────── Models ──────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class AnalyzeRequest(BaseModel):
    url: str
    is_own: bool = False


class GenerateRequest(BaseModel):
    theme: str
    duration_sec: int = 30


class EntryUpdateRequest(BaseModel):
    note: Optional[str] = None
    is_own: Optional[bool] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None


# ────────────────────────────── Helpers ──────────────────────────────

def detect_source(url: str) -> str:
    if "tiktok.com" in url:
        return "tiktok"
    if "instagram.com" in url:
        return "instagram"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "other"


def fetch_metadata(url: str) -> dict:
    result = subprocess.run(
        [str(YT_DLP), "--dump-single-json", "--no-download", "--no-warnings", url],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def extract_metrics(meta: dict) -> dict:
    return {
        "view_count": int(meta.get("view_count") or 0),
        "like_count": int(meta.get("like_count") or 0),
        "comment_count": int(meta.get("comment_count") or 0),
    }


def download_audio(url: str, out_dir: Path) -> Path:
    out_template = str(out_dir / "audio.%(ext)s")
    result = subprocess.run(
        [
            str(YT_DLP),
            "-x", "--audio-format", "mp3", "--audio-quality", "5",
            "--ffmpeg-location", str(FFMPEG_DIR),
            "-o", out_template, url,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"動画の取得に失敗しました: {result.stderr.strip()[-500:]}",
        )
    audio_path = out_dir / "audio.mp3"
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="音声ファイルが生成されませんでした")
    return audio_path


# ────────────────────────────── Insights Engine ──────────────────────────────

PROMPT_INSIGHTS = """あなたはショート動画マーケティングのトップアナリストです。
以下は蓄積された動画の台本データです（再生数順、上位ほど高パフォーマンス）。

# 分析タスク
これらに共通する「勝ちパターン」を抽出してください。観点:
- **フックの型** — 最初の1〜3秒の型（質問・断定・違和感・数字提示など）
- **構成の型** — 起承転結 / 結論先出し / 問題→解決 / Before-After など
- **言葉の使い方** — 口語/敬語、テンポ、平均文字数、繰り返し
- **感情トリガー** — 共感・驚き・憧れ・笑い・恐れのどれか
- **CTA の型** — 質問投げかけ / 保存促し / シリーズ化予告 など
- **再生数の差分** — 上位と下位で何が違うか
- **ジャンルごとの特徴**（複数ジャンルが混在する場合）

# 出力形式（必ずこのJSONで返す）
{
  "report": "Markdown形式の詳細レポート。## セクション分けで読みやすく。具体例を必ず含めること",
  "summary": "今後の台本生成プロンプトに毎回注入する圧縮版。300〜500文字、箇条書きの鉄則リスト形式。「〜すべき」「〜は避ける」など実行可能な指示で書く"
}
"""


def latest_insight_summary() -> Optional[str]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM insights ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return row_dict(row).get("summary")


def generate_insights() -> Optional[dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT id, title, source, uploader, view_count, like_count, comment_count,
               duration, original
        FROM entries
        ORDER BY view_count DESC, id DESC
        LIMIT {PH}
        """,
        (INSIGHT_TOP_N,),
    )
    top = [row_dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) AS c FROM entries")
    total_count = row_dict(cursor.fetchone())["c"]
    conn.close()

    if not top:
        return None

    materials = []
    for i, e in enumerate(top, 1):
        materials.append(
            f"### #{i} [{e['source']}] {e.get('title') or '(無題)'}\n"
            f"- 再生: {e['view_count']:,} / いいね: {e['like_count']:,} / コメント: {e['comment_count']:,} / 尺: {e['duration']}秒\n"
            f"- 投稿者: {e.get('uploader') or '不明'}\n"
            f"- 元台本: {(e['original'] or '')[:400]}"
        )
    prompt = PROMPT_INSIGHTS + "\n\n## 分析対象データ（上位ほど高パフォーマンス）\n\n" + "\n\n".join(materials)

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        print(f"[insights] generation error: {e}")
        return None

    report = (data.get("report") or "").strip()
    summary = (data.get("summary") or "").strip()
    if not report or not summary:
        return None

    total_views = sum(e["view_count"] for e in top)
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO insights (report, summary, entries_analyzed, total_views, created_at)
        VALUES ({PH}, {PH}, {PH}, {PH}, {PH})
        """,
        (report, summary, total_count, total_views, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return {"report": report, "summary": summary, "entries_analyzed": total_count}


def maybe_refresh_insights() -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM entries")
    entry_count = row_dict(cursor.fetchone())["c"]
    cursor.execute("SELECT entries_analyzed FROM insights ORDER BY id DESC LIMIT 1")
    last = cursor.fetchone()
    conn.close()
    last_analyzed = row_dict(last).get("entries_analyzed", 0) if last else 0
    if entry_count >= 1 and (entry_count - last_analyzed) >= INSIGHT_TRIGGER_DELTA:
        threading.Thread(target=generate_insights, daemon=True).start()


# ────────────────────────────── Generation Prompts ──────────────────────────────

def insight_block() -> str:
    summary = latest_insight_summary()
    if not summary:
        return ""
    return (
        "# 🧠 これまでの蓄積から学習した「勝ちパターン」（必ず反映すること）\n\n"
        + summary
        + "\n\n---\n"
    )


def remix_prompt() -> str:
    return (
        insight_block()
        + """あなたはショート動画（TikTok / Instagram Reels）のバズ動画を量産するプロの放送作家です。
以下の音声はショート動画から抽出したものです。

# タスク
1. 音声を **日本語で正確に文字起こし** してください（フィラーも自然に）
2. その台本のスタイル・テーマを踏まえ、**上記の勝ちパターンを必ず反映した改良台本を3パターン** 作成

各パターンは異なる戦略で:
- **フック強化型** — 最初の1〜3秒で指を止めさせるインパクト重視
- **感情訴求型** — 共感・驚き・憧れなど感情を強く動かす
- **意外性/逆張り型** — 常識を覆すギャップで惹きつける

各パターンには次を必ず含めてください（Markdown形式）:
- ## パターン名
- **想定尺**: 〇〇秒
- **タイトル/キャプション案**: ...
- **台本本文**:
  - [0-3秒] フック: ...
  - [3-10秒] 展開: ...
  - [10-25秒] ピーク: ...
  - [25-30秒] オチ/CTA: ...
- **狙い**: 勝ちパターンのどれを採用したか明記し、なぜバズるか2〜3行で

# 出力形式（必ずこのJSONで返す）
{
  "original": "元動画の文字起こし全文",
  "variants": "Markdown形式の3パターンの改良台本"
}
"""
    )


def generate_prompt(theme: str, duration_sec: int) -> str:
    return (
        insight_block()
        + f"""あなたはショート動画のバズ動画を量産するプロの放送作家です。
以下のテーマで、**上記の勝ちパターンを必ず反映した** 伸びる台本を3パターン作ってください。

# テーマ
{theme}

# 想定尺
{duration_sec}秒

# パターン
- **フック強化型** — 最初の1〜3秒のインパクト最大化
- **感情訴求型** — 共感・驚き・憧れを動かす
- **意外性/逆張り型** — ギャップで惹きつける

各パターンには次を必ず含めてください:
- ## パターン名
- **タイトル/キャプション案**: ...
- **台本本文**: 時間区切りで（[0-3秒], [3-10秒], ...）
- **想定サムネ/カバー文言**: ...
- **狙い**: どの勝ちパターンを採用したか + 理由

# 出力形式（必ずこのJSONで返す）
{{
  "scripts": "Markdown形式の3パターンの台本"
}}
"""
    )


# ────────────────────────────── DB ops ──────────────────────────────

def save_entry(
    url: str, source: str, meta: dict, original: str, variants: str,
    is_own: bool, created_by: str,
) -> int:
    metrics = extract_metrics(meta)
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO entries
            (url, source, title, thumbnail, uploader, duration,
             view_count, like_count, comment_count, is_own,
             original, variants, created_by, created_at, metrics_updated_at)
        VALUES ({PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH},{PH})
        RETURNING id
        """,
        (
            url, source,
            meta.get("title", ""), meta.get("thumbnail", ""), meta.get("uploader", ""),
            int(meta.get("duration") or 0),
            metrics["view_count"], metrics["like_count"], metrics["comment_count"],
            1 if is_own else 0,
            original, variants, created_by, now, now,
        ),
    )
    entry_id = row_dict(cursor.fetchone())["id"]
    conn.commit()
    conn.close()
    return int(entry_id)


def get_entry_dict(entry_id: int) -> dict:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM entries WHERE id = {PH}", (entry_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="ナレッジが見つかりません")
    return row_dict(row)


def refresh_entry_metrics(entry_id: int) -> Optional[dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT url FROM entries WHERE id = {PH}", (entry_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    meta = fetch_metadata(row_dict(row)["url"])
    if not meta:
        return None
    metrics = extract_metrics(meta)
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE entries
        SET view_count = {PH}, like_count = {PH}, comment_count = {PH}, metrics_updated_at = {PH}
        WHERE id = {PH}
        """,
        (
            metrics["view_count"], metrics["like_count"], metrics["comment_count"],
            datetime.now().isoformat(timespec="seconds"),
            entry_id,
        ),
    )
    conn.commit()
    conn.close()
    return metrics


# ────────────────────────────── Background worker ──────────────────────────────

def background_metrics_refresher():
    while True:
        try:
            time.sleep(METRICS_REFRESH_INTERVAL)
            now = datetime.now()
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, metrics_updated_at FROM entries
                ORDER BY (CASE WHEN metrics_updated_at IS NULL THEN 0 ELSE 1 END), metrics_updated_at ASC
                LIMIT {PH}
                """,
                (METRICS_REFRESH_BATCH * 5,),
            )
            rows = [row_dict(r) for r in cursor.fetchall()]
            conn.close()

            picked = []
            for r in rows:
                last = r.get("metrics_updated_at")
                if last is None:
                    picked.append(r["id"])
                else:
                    try:
                        last_dt = datetime.fromisoformat(last)
                        if (now - last_dt).total_seconds() > METRICS_AGE_THRESHOLD_SEC:
                            picked.append(r["id"])
                    except Exception:
                        picked.append(r["id"])
                if len(picked) >= METRICS_REFRESH_BATCH:
                    break

            for entry_id in picked:
                try:
                    refresh_entry_metrics(entry_id)
                except Exception as e:
                    print(f"[bg] refresh failed for #{entry_id}: {e}")
        except Exception as e:
            print(f"[bg] worker error: {e}")


threading.Thread(target=background_metrics_refresher, daemon=True).start()


# ────────────────────────────── Auth API ──────────────────────────────

@app.post("/api/login")
def login(req: LoginRequest, response: Response):
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="サーバー設定エラー: APP_PASSWORD 未設定")
    username = (req.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="ユーザー名を入力してください")
    if len(username) > 30:
        raise HTTPException(status_code=400, detail="ユーザー名は30文字以内で")
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="パスワードが違います")
    token = make_token(username)
    response.set_cookie(
        "session", token,
        httponly=True, samesite="lax",
        secure=os.getenv("RENDER", "") != "" or os.getenv("FORCE_HTTPS", "") == "1",
        max_age=60 * 60 * 24 * 30,
    )
    return {"username": username}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/me")
def me(username: str = Depends(current_user)):
    return {"username": username}


# ────────────────────────────── Main API ──────────────────────────────

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, username: str = Depends(current_user)):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URLを入力してください")
    if not YT_DLP.exists():
        raise HTTPException(status_code=500, detail=f"yt-dlp が見つかりません: {YT_DLP}")
    if not FFMPEG.exists():
        raise HTTPException(status_code=500, detail=f"ffmpeg が見つかりません: {FFMPEG}")

    source = detect_source(req.url)
    meta = fetch_metadata(req.url)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = download_audio(req.url, tmp_dir)
        audio_bytes = audio_path.read_bytes()

    if len(audio_bytes) > MAX_INLINE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"動画が長すぎます（音声 {len(audio_bytes) // 1024 // 1024}MB）。短い動画でお試しください。",
        )

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                remix_prompt(),
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API エラー: {e}")

    try:
        data = json.loads(response.text)
        original = (data.get("original") or "").strip()
        variants = (data.get("variants") or "").strip()
    except (json.JSONDecodeError, AttributeError) as e:
        raise HTTPException(status_code=500, detail=f"AIの応答を解析できませんでした: {e}")

    if not original:
        raise HTTPException(status_code=422, detail="音声から文字起こしできませんでした")

    entry_id = save_entry(req.url, source, meta, original, variants, req.is_own, username)
    threading.Thread(target=maybe_refresh_insights, daemon=True).start()
    return get_entry_dict(entry_id)


@app.post("/api/generate")
def generate_blank(req: GenerateRequest, username: str = Depends(current_user)):
    theme = (req.theme or "").strip()
    if not theme:
        raise HTTPException(status_code=400, detail="テーマを入力してください")
    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=generate_prompt(theme, req.duration_sec),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.95,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API エラー: {e}")
    return {
        "theme": theme,
        "duration_sec": req.duration_sec,
        "scripts": (data.get("scripts") or "").strip(),
        "insight_used": bool(latest_insight_summary()),
    }


@app.get("/api/entries")
def list_entries(username: str = Depends(current_user)):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, url, source, title, thumbnail, uploader, duration,
               view_count, like_count, comment_count, is_own, note,
               created_by, created_at, metrics_updated_at
        FROM entries
        ORDER BY id DESC
        """
    )
    rows = [row_dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@app.get("/api/entries/{entry_id}")
def get_entry(entry_id: int, username: str = Depends(current_user)):
    return get_entry_dict(entry_id)


@app.patch("/api/entries/{entry_id}")
def update_entry(entry_id: int, body: EntryUpdateRequest, username: str = Depends(current_user)):
    fields = []
    values = []
    if body.note is not None:
        fields.append(f"note = {PH}"); values.append(body.note)
    if body.is_own is not None:
        fields.append(f"is_own = {PH}"); values.append(1 if body.is_own else 0)
    if body.view_count is not None:
        fields.append(f"view_count = {PH}"); values.append(body.view_count)
    if body.like_count is not None:
        fields.append(f"like_count = {PH}"); values.append(body.like_count)
    if body.comment_count is not None:
        fields.append(f"comment_count = {PH}"); values.append(body.comment_count)
    if not fields:
        return {"ok": True, "updated": 0}
    values.append(entry_id)
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE entries SET {', '.join(fields)} WHERE id = {PH}", values)
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="ナレッジが見つかりません")
    return {"ok": True, "updated": affected}


@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int, username: str = Depends(current_user)):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM entries WHERE id = {PH}", (entry_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="ナレッジが見つかりません")
    return {"ok": True}


@app.post("/api/entries/{entry_id}/refresh-metrics")
def refresh_metrics_endpoint(entry_id: int, username: str = Depends(current_user)):
    metrics = refresh_entry_metrics(entry_id)
    if metrics is None:
        raise HTTPException(status_code=502, detail="メトリクスの再取得に失敗しました")
    return {"ok": True, **metrics}


@app.get("/api/insights")
def get_insights(username: str = Depends(current_user)):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM insights ORDER BY id DESC LIMIT 1")
    current = cursor.fetchone()
    cursor.execute(
        "SELECT id, entries_analyzed, total_views, created_at FROM insights ORDER BY id DESC LIMIT 10"
    )
    history = [row_dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) AS c FROM entries")
    total_entries = row_dict(cursor.fetchone())["c"]
    conn.close()
    last_analyzed = row_dict(current).get("entries_analyzed", 0) if current else 0
    return {
        "current": row_dict(current) if current else None,
        "history": history,
        "total_entries": total_entries,
        "next_refresh_in": max(0, INSIGHT_TRIGGER_DELTA - (total_entries - last_analyzed)),
    }


@app.post("/api/insights/refresh")
def refresh_insights_endpoint(username: str = Depends(current_user)):
    result = generate_insights()
    if result is None:
        raise HTTPException(status_code=400, detail="ナレッジが不足しています（最低1件必要）")
    return {"ok": True, **result}


# ────────────────────────────── Static ──────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")
