import requests
r = requests.post('https://api.supabase.com/v1/projects/cxpablqwnwvacuvhcjen/database/query',
    headers={'Authorization': 'Bearer sbp_a6ec957deec4b92d6453e7fbdf723d9be6a5f78e', 'Content-Type': 'application/json'},
    json={'query': "SELECT * FROM bot_status ORDER BY bot_name"})
for row in r.json():
    print(row)
