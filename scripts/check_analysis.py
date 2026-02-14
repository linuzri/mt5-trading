import json, os
from datetime import datetime

# Check JSON file
data_dir = 'vercel-dashboard/data'
path = f'{data_dir}/daily_analysis.json'
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        dates = sorted(data.keys())
    elif isinstance(data, list):
        dates = sorted(set(d.get('date','') for d in data if d.get('date')))
    print(f"JSON dates (last 5): {dates[-5:]}")
    print(f"Latest JSON: {dates[-1]}")
else:
    print("No daily_analysis.json found")

# Check Supabase
try:
    from dotenv import load_dotenv
    load_dotenv()
    import requests
    url = os.getenv("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
    key = os.getenv("SUPABASE_KEY")
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    r = requests.get(f"{url}/rest/v1/daily_analysis?select=date&order=date.desc&limit=5", headers=headers)
    rows = r.json()
    print(f"\nSupabase latest dates: {[row['date'] for row in rows]}")
except Exception as e:
    print(f"\nSupabase check failed: {e}")

# Check cron - when was last digest?
print(f"\nToday: {datetime.now().strftime('%Y-%m-%d')}")
