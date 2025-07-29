import React from 'react';

function Modal({ show, message, onClose }) {
  if (!show) return null;
  return (
    <div className="modal fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="modal-content bg-white p-8 rounded-xl shadow-lg w-4/5 max-w-md text-center">
        <p>{message}</p>
        <button
          className="bg-indigo-700 text-white px-4 py-2 rounded-lg mt-4"
          onClick={onClose}
        >
          OK
        </button>
      </div>
    </div>
  );
}

export default Modal; 