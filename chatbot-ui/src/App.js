import React, { useState, useRef } from 'react';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import LoadingIndicator from './components/LoadingIndicator';
import ChatInput from './components/ChatInput';
import Modal from './components/Modal';
import axios from 'axios';

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [message, setMessage] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [websiteLink, setWebsiteLink] = useState('');
  const [loading, setLoading] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [showModal, setShowModal] = useState(false);
  const chatMessagesRef = useRef();

  React.useEffect(() => {
    async function fetchData() {
      const response = await getChatHistory("1093828", "238291");
      setChatHistory([...response]);
    }

    fetchData();
  }, []);

  React.useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const addMessage = (role, content) => {
    setChatHistory(prev => [...prev, { role, content }]);
  };                                                                               

  async function getChatHistory(userId, sessionId) {
    try {
      const response = await axios.get(`https://f8c14eefce75.ngrok-free.app/chat-history?userId=${userId}&sessionId=${sessionId}`, {
        headers: {
          "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
        }
      })
      console.log(response);
      const chat_history = await JSON.parse(response.data.history);
      return chat_history
    } catch (error) {
      console.log("------>>>>>error", error)
      return error.res
    }
  }

  async function sendMessage(message) {
    try {
      console.log('----->>>>>>>>>', message);
      const response = await axios.post(
        "https://f8c14eefce75.ngrok-free.app/chat",
        { question: message, userId: "1093828", sessionId: "238291", headers: {
          "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
        } }
      );
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  } 

  const handleMessage = async (message) => {
    try {
      const response = await sendMessage(message);
      return response.answer;
    } catch (error) {
      console.error(error);
    }
  }

  const handleSend = async () => {
    const hasFiles = selectedFiles.length > 0;
    const hasLink = websiteLink.trim() !== '';
    if (!message && !hasFiles && !hasLink) {
      setModalMessage('Please type a message, select files, or enter a website link.');
      setShowModal(true);
      return;
    }
    if (message) {
      addMessage('user', message);
      setLoading(true);
      setTimeout(() => {
        const answer = handleMessage(message);
        addMessage('ai', answer);
        setLoading(false);
      }, 1500);
      setMessage('');
    }
    
    
    // if (hasFiles) {
    //   setLoading(true);
    //   setModalMessage('Uploading files...');
    //   setShowModal(true);
    //   setTimeout(() => {
    //     setModalMessage(`Successfully uploaded and processed ${selectedFiles.length} file(s).`);
    //     setSelectedFiles([]);
    //     setLoading(false);
    //   }, 2000);
    // }
    // if (hasLink) {
    //   setLoading(true);
    //   setModalMessage('Ingesting website link...');
    //   setShowModal(true);
    //   setTimeout(() => {
    //     setModalMessage(`Successfully ingested link: ${websiteLink}`);
    //     setWebsiteLink('');
    //     setLoading(false);
    //   }, 2000);
    // }
  };

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleRemoveFile = (index) => {
    setSelectedFiles(files => files.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 font-sans">
      <div className="chat-container flex flex-col w-full max-w-2xl h-[90vh] max-h-[900px] bg-white rounded-2xl shadow-xl overflow-hidden">
        <ChatHeader />
        <MessageList chatHistory={chatHistory} ref={chatMessagesRef} />
        {loading && <LoadingIndicator />}
        <ChatInput
          message={message}
          setMessage={setMessage}
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
          websiteLink={websiteLink}
          setWebsiteLink={setWebsiteLink}
          onSend={handleSend}
          onFileChange={handleFileChange}
          onRemoveFile={handleRemoveFile}
        />
        <Modal show={showModal} message={modalMessage} onClose={() => setShowModal(false)} />
      </div>
    </div>
  );
}

export default App;
