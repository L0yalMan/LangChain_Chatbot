import React, { useState } from 'react';
import axios from 'axios';
import MultipleFileUpload from './multipleFileUpload';

function ChatInput({
  message,
  setMessage,
  selectedFiles,
  setSelectedFiles,
  websiteLink,
  setWebsiteLink,
  onSend,
  onFileChange,
  onRemoveFile
}) {
  const [localWebsiteLink, setLocalWebsiteLink] = useState("");

  async function uploadDatasetFile(formData) {
    try {
      const response = await axios.post(
        "https://56ab6a3bf149.ngrok-free.app/files/upload",
        formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
          }
        }
      );
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  } 

  const handleUploadFile = async () => {
    try {
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', '1093828')
        const response = await uploadDatasetFile(formData);
        console.log(response);
      }
    } catch (error) {
      console.error(error);
    }
  }

  const handleIngestLink = async () => {
    try {
      const response = await axios.post(
        "https://56ab6a3bf149.ngrok-free.app/website/ingest",
        {
          userId: "1093828",
          url: localWebsiteLink,
          headers: {
            "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
          }
        }
      )

      console.log(response);
    } catch (error) {
      console.error(error);
    }
  }
  
  return (
    <div className="chat-input-area px-6 py-4 border-t border-gray-200 bg-white flex flex-col gap-3">
      <div className="chat-input-row flex gap-3 items-center">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          className="flex-grow p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
          rows={2}
          placeholder="Type your message..."
        />
        <button
          onClick={onSend}
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105"
        >
          Send
        </button>
      </div>
      <div className="file-upload-area flex flex-wrap gap-2 items-center text-sm text-gray-600">
        <MultipleFileUpload 
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
          handleUploadFile={handleUploadFile}
        />
        <input
          type="text"
          value={localWebsiteLink}
          onChange={e => setLocalWebsiteLink(e.target.value)}
          className="flex-grow p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
          placeholder="Or paste a website link here..."
        />
        <button
          type="button"
          onClick={() => {
            if (localWebsiteLink.trim()) handleIngestLink()
          }}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg shadow-sm transition duration-300 ease-in-out transform hover:scale-105"
        >
          Ingest Link
        </button>
      </div>
    </div>
  );
}

export default ChatInput; 