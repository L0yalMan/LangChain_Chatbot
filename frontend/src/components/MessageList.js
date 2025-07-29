import React, { forwardRef } from 'react';

const MessageList = forwardRef(({ chatHistory }, ref) => (
  <div ref={ref} className="chat-messages flex-1 p-6 overflow-y-auto flex flex-col gap-4 bg-gray-50">
    {chatHistory.map((msg, idx) => (
      <div
        key={idx}
        className={`message-bubble max-w-[75%] px-4 py-3 rounded-xl break-words ${msg.role === 'user' ? 'self-end bg-purple-400 text-white rounded-br-md' : 'self-start bg-indigo-100 text-gray-800 rounded-bl-md'}`}
      >
        {msg.content}
      </div>
    ))}
  </div>
));

export default MessageList; 