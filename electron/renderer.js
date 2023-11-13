let peerConnection = new RTCPeerConnection();

let stream = await navigator.mediaDevices.getUserMedia({
  audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  },
  video: false,
})
let audioTracks = stream.getAudioTracks();
if (audioTracks.length > 0) {
    console.log('Using Audio device: ' + audioTracks[0].label);
}
stream.oninactive = function() {
    console.log('Stream ended');
};
let localStream = stream;

function handleError(error) {
  console.log('navigator.getUserMedia error: ', error);
}

let ws = new WebSocket('ws://localhost:8089');

let opened = new Promise((resolve, reject) => {
    ws.onopen = () => {
        console.log('Connected to the signaling server');
        resolve();
    };
});
ws.onerror = err => {
  console.error(err);
};

ws.onmessage = msg => {
  let data = JSON.parse(msg.data);

  switch(data.type) {
    case 'offer':
      console.log(data);
      peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
      peerConnection.createAnswer()
        .then(answer => {
          peerConnection.setLocalDescription(new RTCSessionDescription(answer));
          ws.send(JSON.stringify({ type: 'answer', answer: answer }));
        })
        .catch(error => {
          console.error('Error when creating an answer', error);
        });
      break;
    case 'answer':
      console.log(data);
      peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
      break;
    case 'candidate':
      let candidate = new RTCIceCandidate({
        sdpMLineIndex: data.label,
        candidate: data.candidate
      });
      peerConnection.addIceCandidate(candidate);
      break;
    default:
      break;
  }
};

await opened;
peerConnection.onicecandidate = event => {
  if (event.candidate) {
    ws.send(JSON.stringify({
      type: 'candidate',
      label: event.candidate.sdpMLineIndex,
      id: event.candidate.sdpMid,
      candidate: event.candidate.candidate
    }));
  }
};
peerConnection.onconnectionstatechange = event => {
  console.log('Connection state change', event);
};

peerConnection.onaddstream = event => {
  let audio = new Audio();
  audio.srcObject = event.stream;
  audio.onloadedmetadata = function(e) {
    audio.play();
  };
};

localStream.getTracks().forEach(track => {
  peerConnection.addTrack(track, localStream);
});

peerConnection.createOffer()
  .then(offer => {
    ws.send(JSON.stringify({ type: 'offer', offer: offer }));
    peerConnection.setLocalDescription(new RTCSessionDescription(offer));
  })
  .catch(error => {
    console.error('Error when creating an offer', error);
  });

