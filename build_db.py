from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Baca PDF
reader = PdfReader("SBS10022_Sains.pdf")

text = ""

for page in reader.pages:
    content = page.extract_text()
    if content:
        text += content

# Pecahkan kepada chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Bilangan chunk:", len(chunks))

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Simpan database
db = FAISS.from_texts(chunks, embeddings)

db.save_local("sbs_db")

print("Database berjaya dibina!")
