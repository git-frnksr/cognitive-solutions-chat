# Cognitive Solutions - Base Chat

## Chat with your files

## Running locally

1. Install dependencies: pip install -r requirements.txt
2. Copy .env.example to .env file and update your key
3. Create a folder named 'in' and a folder named 'db' in your solution
4. Add a pdf file into in folder and update the path in the ingest.py
4. In a terminal window run python ingest.py. This will create an embedding file for your imported pdf
5. Run python main.py. This will start a web server on port 9000. You can now access the bot using localhost:9000 with your browser
