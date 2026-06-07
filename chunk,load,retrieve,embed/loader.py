import os 
from pypdf import PdfReader
import json
import requests


class Document:
    page_content: str
    metadata: dict
class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        if not self.file_path.endswith(".txt"):
            raise ValueError("Only .txt files are supported")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        with open(
            self.file_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read()
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "file_type": "txt",
                    "file_size": os.path.getsize(self.file_path)
                }
            )
        ]
class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        if not self.file_path.endswith(".pdf"):
            raise ValueError("Only .pdf files are supported")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        reader = PdfReader(self.file_path)
        documents = []
        for page in reader.pages:
            text = page.extract_text()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "file_type": "pdf",
                        "file_size": os.path.getsize(self.file_path)
                    }
                )
            )
        return documents
class CSVLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        if not self.file_path.endswith(".csv"):
            raise ValueError("Only .csv files are supported")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        with open(
            self.file_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read()
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "file_type": "csv",
                    "file_size": os.path.getsize(self.file_path)
                }
            )
        ]
class JSONLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def load(self):
        if not self.file_path.endswith(".json"):
            raise ValueError("Only .json files are supported")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        with open(
            self.file_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = json.load(file)
        return [
            Document(
                page_content=str(text),
                metadata={
                    "source": self.file_path,
                    "file_type": "json",
                    "file_size": os.path.getsize(self.file_path)
                }
            )
        ]
class URLLoader:
    def __init__(self, url: str):
        self.url = url

    def load(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch URL: {self.url}")
        text = response.text
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.url,
                    "file_type": "url",
                    "file_size": len(text)
                }
            )
        ]
class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str):
        if file_path.endswith(".txt"):
            return TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            return PDFLoader(file_path)
        elif file_path.endswith(".csv"):
            return CSVLoader(file_path)
        elif file_path.endswith(".json"):
            return JSONLoader(file_path)
        elif file_path.startswith("http://") or file_path.startswith("https://"):
            return URLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    