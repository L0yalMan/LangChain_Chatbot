import React from 'react';

function ChatHeader() {
  return (
    <div className="chat-header px-6 py-4 bg-indigo-700 text-white rounded-t-2xl flex justify-between items-center">
      <h1 className="text-xl font-semibold">AI Assistant</h1>
      <span className="text-sm">Guest User</span>
    </div>
  );
}

export default ChatHeader; 