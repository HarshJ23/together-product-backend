/// api
import express from "express";
import bodyParser from "body-parser";
import cors from "cors"; 
import 'dotenv/config';
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TokenTextSplitter } from "langchain/text_splitter";
import { SearchApiLoader } from "langchain/document_loaders/web/searchapi";

const app = express();
const port = 4000; // You can change this to the desired port

app.use(bodyParser.json());
app.use(cors());

// api keys
// const openai_key = process.env['OPENAI_API_KEY']
// const apiKey = process.env['SEARCH_API_KEY']


app.post("/api/analyze", async (req, res) => {
  try {
    const openai_key = process.env.OPENAI_API_KEY;
    const apiKey = process.env.SEARCH_API_KEY;

    const llm = new OpenAI();
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: openai_key,
      batchSize: 512,
    });

    const {question,query} = req.body;
    // const question = 'Give pros and cons by analyzing the specs and features of the mobile given in query';

    const loader = new SearchApiLoader({ q: query, apiKey, engine: "google" });
    const docs = await loader.load();

    const textSplitter = new TokenTextSplitter({
      chunkSize: 800,
      chunkOverlap: 100,
    });

    const splitDocs = await textSplitter.splitDocuments(docs);

    const vectorStore = await MemoryVectorStore.fromDocuments(
      splitDocs,
      embeddings
    );


    const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), {
      verbose: true,
    });

    
    const answer = await chain.call({ query: question });

    res.json({ answer: answer.text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});



app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

