from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import httpx
import json
import tiktoken
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table, select, func, ForeignKey, inspect, DateTime, UniqueConstraint, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize tokenizer for length checks
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://jfk-arena.fly.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize battle statistics
battle_stats: Dict[str, dict] = {}

# ELO rating constant
K_FACTOR = 32

# Initialize RAG components
try:
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "cache")
    cache_path = os.path.join(cache_dir, "faiss_store")
    data_dir = os.path.join(base_dir, "data", "jfk_text")
    
    print(f"Cache path: {cache_path}")
    print(f"Data directory: {data_dir}")
    
    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(cache_path):
        print("Loading vector store from cache...")
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.load_local(cache_path, embeddings)
            print("Successfully loaded vector store from cache!")
        except Exception as e:
            print(f"Failed to load from cache: {str(e)}")
            print("Will attempt to recreate vector store...")
            os.remove(cache_path)  # Remove corrupted cache
            raise  # Trigger the recreation logic
    
    if not os.path.exists(cache_path):
        # If cache doesn't exist or failed to load, create new vector store
        print("Creating new vector store...")
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory not found: {data_dir}")
            print("Creating empty data directory...")
            os.makedirs(data_dir, exist_ok=True)
            
        loader = DirectoryLoader(data_dir, glob="**/*.md", show_progress=True)
        print(f"Loading documents from {data_dir}...")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        if not documents:
            print("Warning: No documents found in data directory")
            vectorstore = None
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            print("Splitting documents into chunks...")
            texts = text_splitter.split_documents(documents)
            print(f"Created {len(texts)} text chunks")
            
            print("Initializing embeddings and vector store...")
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # Save to cache
            os.makedirs(cache_dir, exist_ok=True)
            vectorstore.save_local(cache_path)
            print("Successfully created and cached vector store!")

except Exception as e:
    print(f"Warning: Failed to initialize RAG components: {str(e)}")
    print("Please ensure the data/jfk_text directory exists and contains .md files")
    vectorstore = None

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable is not set")

print(f"Connecting to database: {DATABASE_URL}")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Automatically check if connection is valid
    pool_recycle=3600,   # Connection recycle time (1 hour)
    pool_size=5,         # Connection pool size
    max_overflow=10,     # Maximum overflow connections
    pool_timeout=30,     # Connection timeout
    connect_args={
        "connect_timeout": 10,  # Connection timeout
        "keepalives": 1,        # Enable keepalive
        "keepalives_idle": 30,  # Keepalive idle time
        "keepalives_interval": 10,  # Keepalive interval
        "keepalives_count": 5,   # Keepalive retry count
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database tables
metadata = MetaData()

user = Table(
    "user",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("email", String, nullable=False, unique=True),
    Column("email_verified", Boolean, nullable=False),
    Column("image", String, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    Column("anonymous", Boolean, nullable=False, default=False),
)

models = Table(
    "models",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("model_id", String),
    Column("name", String),
    Column("wins", Integer, default=0),
    Column("losses", Integer, default=0),
    Column("draws", Integer, default=0),
    Column("invalid", Integer, default=0),
    Column("elo", Float, default=1500.0),
    Column("user_id", String, ForeignKey("user.id", ondelete="CASCADE"), nullable=True),
    UniqueConstraint('user_id', 'model_id', name='models_user_id_model_id_key')
)

battles = Table(
    "battles",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("model1_id", Integer, ForeignKey("models.id", ondelete="CASCADE")),
    Column("model2_id", Integer, ForeignKey("models.id", ondelete="CASCADE")),
    Column("winner_id", Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=True),
    Column("question", String),
    Column("response1", String),
    Column("response2", String),
    Column("result", String, nullable=True),  # 'model1_win', 'model2_win', 'draw', 'invalid'
    Column("created_at", DateTime, default=func.now()),
    Column("voted_at", DateTime, nullable=True),
    Column("user_id", String, ForeignKey("user.id", ondelete="CASCADE"), nullable=True),
)

# Supported model list
SUPPORTED_MODELS = [
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
    {"id": "openai/o1", "name": "gpt-o1"},
    {"id": "openai/gpt-4o-2024-11-20", "name": "GPT-4o"},
    {"id": "openai/gpt-4", "name": "GPT-4"},
    {"id": "google/gemini-2.0-flash-exp:free", "name": "Gemini 2.0 Flash"},
    {"id": "google/gemini-2.5-pro-exp-03-25:free", "name": "Gemini 2.5 Pro"},
    {"id": "qwen/qwen2.5-32b-instruct", "name": "qwen2.5-32b-instruct"},
    {"id": "qwen/qwen-2.5-72b-instruct", "name": "qwen2.5-72b-instruct"},
    {"id": "anthropic/claude-3.7-sonnet", "name": "Claude 3.7 Sonnet"},
    {"id": "anthropic/claude-3.5-haiku-20241022:beta", "name": "Claude 3.5 haiku"},
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 sonnet"},
    {"id": "nvidia/llama-3.1-nemotron-70b-instruct:free", "name": "Nvidia Llama 3.1 70B"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B"},
    {"id": "mistralai/mistral-small-3.1-24b-instruct:free", "name": "Mixtral 3.1 24B"},
    {"id": "mistralai/mistral-small-24b-instruct-2501:free", "name": "Mistral 3 24B"},
    {"id": "deepseek/deepseek-chat-v3-0324:free", "name": "DeepSeek v3 0324"},
    {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek r1"},
]

# Test database connection
def test_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Successfully connected to the database!")
        return True
    except Exception as e:
        print(f"Error connecting to the database: {str(e)}")
        return False

# Check if tables exist
def check_tables_exist():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print(f"Existing tables: {existing_tables}")
    return "models" in existing_tables and "battles" in existing_tables

# Initialize models in the database
def init_models():
    with SessionLocal() as db:
        try:
            for model in SUPPORTED_MODELS:
                # Check if model exists
                result = db.execute(
                    select(models).where(models.c.model_id == model["id"])
                ).first()
                if not result:
                    # If model doesn't exist, add it
                    db.execute(
                        models.insert().values(
                            model_id=model["id"],
                            name=model["name"],
                            wins=0,
                            losses=0,
                            draws=0,
                            invalid=0,
                            elo=1500.0
                        )
                    )
            db.commit()
            print("Models initialized successfully!")
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            db.rollback()
            raise

# Initialize database
def initialize_database():
    try:
        # Drop only our application tables if they exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        # Only drop tables that we manage
        tables_to_drop = ['models', 'battles']
        for table_name in tables_to_drop:
            if table_name in existing_tables:
                metadata.tables[table_name].drop(bind=engine)
                print(f"Dropped table: {table_name}")
        
        # Create our application tables
        metadata.create_all(bind=engine)
        print("Created new tables")
        
        # Initialize models
        init_models()
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

# Test connection and initialize database at startup
print("Starting database initialization...")
if test_db_connection():
    if not check_tables_exist():
        print("Tables do not exist. Creating tables...")
        if not initialize_database():
            raise Exception("Failed to initialize database")
    else:
        print("Tables already exist. Skipping initialization.")
else:
    raise Exception("Failed to connect to the database")

def update_elo(winner_elo: float, loser_elo: float) -> tuple[float, float]:
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    
    new_winner_elo = winner_elo + K_FACTOR * (1 - expected_winner)
    new_loser_elo = loser_elo + K_FACTOR * (0 - expected_loser)
    
    return new_winner_elo, new_loser_elo

class BattleRequest(BaseModel):
    model1: str
    model2: str
    question: str

    @validator("question")
    def validate_question_length(cls, v):
        tokens = count_tokens(v)
        if tokens > 500:
            raise ValueError(f"Question exceeds 500 tokens (current: {tokens})")
        return v

class VoteRequest(BaseModel):
    result: str  # "model1", "model2", "draw", "invalid"
    model1: str
    model2: str
    question: str

def get_relevant_context(question: str) -> str:
    if vectorstore is None:
        return ""
    try:
        docs = vectorstore.similarity_search(question, k=3)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

async def get_model_response(client: httpx.AsyncClient, model_id: str, question: str) -> str:
    # Get relevant context from RAG
    context = get_relevant_context(question)
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "https://github.com/OpenRouterStudio/openrouter-py",
        "X-Title": "JFK Battle Arena",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are an AI assistant participating in a battle arena. 
    Your task is to provide the most helpful, accurate, and well-reasoned response to the user's question in a concise manner, using a single natural paragraph without bullet points. 
    If relevant context is provided, incorporate it to inform your answer without merely repeating it; 
    instead, synthesize the information and offer a thoughtful answer. 
    Your response must not exceed 1000 tokens, and you should reply in English and plain text."""

    user_prompt = f"""Context: {context}

Question: {question}"""
    
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 2000
    }
    
    try:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120.0
        )
        
        if not response.is_success:
            error_content = response.text
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRouter API error for {model_id}: {error_content}"
            )
            
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"]
        
        # Check response length
        if count_tokens(response_text) > 2000:
            response_text = response_text[:response_text.rindex(" ", 0, 2000)] + "..."
            
        return response_text
        
    except httpx.TimeoutException as e:
        print(f"Timeout error for {model_id}: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail=f"Request timeout for {model_id}"
        )
    except httpx.RequestError as e:
        print(f"Request error for {model_id}: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Request failed for {model_id}: {str(e)}"
        )
    except Exception as e:
        print(f"Unexpected error for {model_id}: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response content: {response.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get response from {model_id}: {str(e)}"
        )

@app.get("/models")
async def get_models():
    return SUPPORTED_MODELS

@app.post("/battle")
async def battle(request: dict):
    model1_id = request.get("model1")
    model2_id = request.get("model2")
    question = request.get("question")

    if not model1_id or not model2_id or not question:
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout to 120 seconds
            # Get responses concurrently
            response1_task = get_model_response(client, model1_id, question)
            response2_task = get_model_response(client, model2_id, question)
            
            try:
                response1, response2 = await asyncio.gather(response1_task, response2_task)
            except Exception as e:
                print(f"Error getting model responses: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get responses from one or both models. Please try again."
                )
            
            # Store battle in database with timeout
            try:
                with SessionLocal() as db:
                    # Get model IDs from database
                    model1 = db.execute(
                        select(models).where(models.c.model_id == model1_id)
                    ).first()
                    model2 = db.execute(
                        select(models).where(models.c.model_id == model2_id)
                    ).first()

                    if not model1 or not model2:
                        raise HTTPException(status_code=400, detail="Invalid model ID")

                    # Create new battle record
                    battle_result = db.execute(
                        battles.insert().values(
                            model1_id=model1.id,
                            model2_id=model2.id,
                            winner_id=None,
                            question=question,
                            response1=response1,
                            response2=response2,
                            result=None,
                            created_at=func.now(),
                            voted_at=None
                        )
                    )
                    db.commit()
                    battle_id = battle_result.inserted_primary_key[0]
                
                return {
                    "battle_id": battle_id,
                    "response1": response1,
                    "response2": response2
                }
            except Exception as e:
                print(f"Database error in battle endpoint: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to store battle results. Please try again."
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again with a shorter question."
        )
    except Exception as e:
        print(f"Unexpected error in battle endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )

@app.post("/vote")
async def vote(request: dict):
    result = request.get("result")
    model1_id = request.get("model1")
    model2_id = request.get("model2")
    battle_id = request.get("battle_id")
    
    if not result or not model1_id or not model2_id or not battle_id:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    with SessionLocal() as db:
        try:
            # Get current status of both models
            model1 = db.execute(
                select(models).where(models.c.model_id == model1_id)
            ).first()
            model2 = db.execute(
                select(models).where(models.c.model_id == model2_id)
            ).first()
            
            if not model1 or not model2:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update battle record with result
            battle_result = None
            winner_id = None
            if result == "model1":
                winner_id = model1.id
                battle_result = "model1_win"
            elif result == "model2":
                winner_id = model2.id
                battle_result = "model2_win"
            elif result == "draw":
                battle_result = "draw"
            else:  # invalid
                battle_result = "invalid"

            # Update battle record
            db.execute(
                battles.update()
                .where(battles.c.id == battle_id)
                .values(
                    winner_id=winner_id,
                    result=battle_result,
                    voted_at=func.now()  # Use func.now() for timestamp
                )
            )
            
            # Update statistics
            if result == "model1":
                # Update win/loss counts
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(wins=models.c.wins + 1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(losses=models.c.losses + 1)
                )
                
                # Calculate new ELO scores
                r1 = 10 ** (model1.elo / 400)
                r2 = 10 ** (model2.elo / 400)
                e1 = r1 / (r1 + r2)
                e2 = r2 / (r1 + r2)
                
                new_elo1 = model1.elo + K_FACTOR * (1 - e1)
                new_elo2 = model2.elo + K_FACTOR * (0 - e2)
                
                # Update ELO scores
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(elo=new_elo1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(elo=new_elo2)
                )
                
            elif result == "model2":
                # Update win/loss counts
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(wins=models.c.wins + 1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(losses=models.c.losses + 1)
                )
                
                # Calculate new ELO scores
                r1 = 10 ** (model1.elo / 400)
                r2 = 10 ** (model2.elo / 400)
                e1 = r1 / (r1 + r2)
                e2 = r2 / (r1 + r2)
                
                new_elo1 = model1.elo + K_FACTOR * (0 - e1)
                new_elo2 = model2.elo + K_FACTOR * (1 - e2)
                
                # Update ELO scores
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(elo=new_elo1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(elo=new_elo2)
                )
                
            elif result == "draw":
                # Update draw counts
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(draws=models.c.draws + 1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(draws=models.c.draws + 1)
                )
                
            elif result == "invalid":
                # Update invalid counts
                db.execute(
                    models.update()
                    .where(models.c.model_id == model1_id)
                    .values(invalid=models.c.invalid + 1)
                )
                db.execute(
                    models.update()
                    .where(models.c.model_id == model2_id)
                    .values(invalid=models.c.invalid + 1)
                )
            
            db.commit()
            return {"status": "success"}
        except Exception as e:
            print(f"Error in vote endpoint: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def get_leaderboard():
    with SessionLocal() as db:
        # Get all models sorted by ELO score in descending order
        result = db.execute(
            select(models).order_by(models.c.elo.desc())
        ).fetchall()
        
        return [
            {
                "id": row.model_id,
                "name": row.name,
                "wins": row.wins,
                "losses": row.losses,
                "draws": row.draws,
                "invalid": row.invalid,
                "elo": row.elo
            }
            for row in result
        ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
