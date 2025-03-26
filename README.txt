command to run chromadb
nohup chroma run --host 0.0.0.0 --port 8000 > chroma.log 2>&1 &


for backend
nohup python backend/api.py > backend.log 2>&1 &

for frontend
pm2 start npm --name "ai-frontend" -- run dev

