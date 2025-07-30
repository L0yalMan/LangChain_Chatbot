import React, { useState, useRef } from 'react';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import LoadingIndicator from './components/LoadingIndicator';
import ChatInput from './components/ChatInput';
import Modal from './components/Modal';
import axios from 'axios';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [websiteLink, setWebsiteLink] = useState('');
  const [loading, setLoading] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [showModal, setShowModal] = useState(false);
  const chatMessagesRef = useRef();                                                                             

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
