from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_url(book_id):
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


# fetch ebook and split into chunks (docs)
def get_chunks(url):
    loader = GutenbergLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=300,
        separators=["\r\n\n\n\r\n\n\n", "\r\n\n\n", "."],
    )

    text = loader.load()[0].page_content

    # remove PROJECT GUTENBERG header and footer sections
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_index = text.find(start_marker)
    start_end_index = text.find("***", start_index + len(start_marker))
    end_index = text.find(end_marker)
    text = text[start_end_index + 3 : end_index]

    # splitting
    docs = text_splitter.create_documents([text])
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("\r\n\n\n", " ")
    print("created %d chunks." % len(docs))
    return docs
