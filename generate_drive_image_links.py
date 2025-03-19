from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

# Setup
FOLDER_ID = '1350jQz7F_roKRRwfmCwh_8x_FJN_Ivuh'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'credentials.json'  # Your downloaded credentials file

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=creds)

# Pagination loop
image_links = {}
page_token = None

while True:
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false",
        pageSize=1000,
        fields="nextPageToken, files(id, name)",
        pageToken=page_token
    ).execute()

    files = results.get('files', [])
    for file in files:
        image_links[file['name']] = f"https://drive.google.com/uc?export=view&id={file['id']}"

    page_token = results.get('nextPageToken', None)
    if page_token is None:
        break

# Save the final mapping
with open('image_link_mapping.json', 'w') as f:
    json.dump(image_links, f, indent=2)

print(f"✅ {len(image_links)} image links generated and saved to image_link_mapping.json")
