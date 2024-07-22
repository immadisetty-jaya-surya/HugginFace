import { useState } from "react"
import {AutoTokenizer,AutoModelForCausalLM} from '@xenova/transformers'

const LoadModel = () => {
  const [messages,setMessages] = useState([
    {
      role:'user',
      content: 'List the steps to bake a chocolate cake from scratch.'
    }
  ]);
  const [response,setResponse] = useState('');
  const [model, setModel] = useState(null);
  const [tokenizer, setTokenizer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tokensPerSecond, setTokensPerSecond] = useState(null);
  const [loadingPercentage, setLoadingPercentage] = useState(0);
  const totalSizeMB = 368.74;

  const loadModel = async () => {
    setLoading(true);
    setLoadingPercentage(0);

    /* const interval = setInterval(() => {
      setLoadingPercentage(prev =>{
        if (prev >= 100) {
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      })
    }, 100); */

    try{
      const loadedTokenizer = await AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")
      const loadedModel = await AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct",{
        progress_callback : (progress) => {
          setLoadingPercentage(Math.round(progress * 100));
        }
      })
      
        // clearInterval(interval);
        setTokenizer(loadedTokenizer);
        setModel(loadedModel);
        setLoadingPercentage(100);
        setLoading(false);
        // clearInterval(interval);
    }catch (error) {
      console.log('Error loading model or tokenizer:', error);
      // clearInterval(interval);
      setLoading(false);
      setLoadingPercentage(0);
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!tokenizer || !model){
      console.log('Tokenizer or model not loaded',tokenizer,model);
      return;
    }

    try{
      const inputText = messages[0].content;
      const input_ids = tokenizer.encode(inputText, {return_tensors: 'pt'});
  
      const startTime = Date.now();
      const output = await model.generate(input_ids,{ 
        max_new_tokens: 50, 
        temperature: 0.6, 
        top_p: 0.92, 
        do_sample: true 
      });
      const endTime = Date.now();
  
      const outputText = tokenizer.decode(output[0]);
      // const outputText = tokenizer.decode(output[0],{skip_special_tokens: true});
      console.log(outputText);
      setResponse(outputText);
  
      const tokensGenerated = output[0].length;
      const timeTaken = (endTime-startTime) / 1000;
      setTokensPerSecond(tokensGenerated/timeTaken);
    }catch(error){
      console.error('Error generating response:', error);
    }
  }

  const handleReset = () => {
    setMessages([{role: 'user',content:''}])
    setResponse('');
    setTokensPerSecond(null)
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-4">SmolLM WebGPU AI</h1>
      <button
        onClick={loadModel}
        className={`w-full max-w-md mb-4 p-2 rounded ${loading || model ? 'bg-gray-400' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
        disabled={loading || model}
      >
        {loading ? `Loading... ${(totalSizeMB * (loadingPercentage / 100)).toFixed(2)}mb of ${totalSizeMB}mb (${loadingPercentage}%)` : 'Load model'}
      </button>
      {
        model && !loading && (
          <div className="w-full max-w-md mb-4 p-2 text-green-500 text-center">
            Model loaded ready to chat
          </div>
        )
      }
      <form onSubmit={handleSubmit} className="w-full max-w-md">
        <textarea 
          className="w-full p-2 mb-4 border border-gray-300 rounded"
          value={messages[0].content}
          onChange={(e) => setMessages([{role:'user',content:e.target.value}])}
        />
        <button 
          type="submit"
          // onSubmit={handleSubmit}
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 mb-2"
        >
          Submit
        </button>
        <button
        type="button"
          onClick={handleReset}
          className="w-full bg-red-500 text-white p-2 rounded hover:bg-red-600"
        >
          Reset chat
        </button>
        <div>
          {response && (
            <div className="w-full max-w-md bg-white p-4 mt-4 border border-gray-300 rounded">
              <h2 className="text-xl font-bold">Response</h2>
              <p>{response}</p>
              {tokensPerSecond && (
                <p className="mt-2">{tokensPerSecond.toFixed(2)} tokens/second</p>
              )}
            </div>
          )}
        </div>
      </form>
    </div>
  )
}

export default LoadModel