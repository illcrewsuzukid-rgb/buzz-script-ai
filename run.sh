#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH="$PWD/bin:$PATH"

if [ ! -f .env ]; then
  echo "❌ .env ファイルがありません。.env.example をコピーしてAPIキーを設定してください。"
  exit 1
fi

echo "🚀 TikTok 台本ジェネレーターを起動します..."
echo "   ブラウザで http://127.0.0.1:8000 を開いてください"
echo ""
exec ./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
