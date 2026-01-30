import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Reserved key for transmitting action_prefix through the websocket protocol.
_ACTION_PREFIX_KEY = "__rtc_action_prefix"


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _send_and_recv(self, payload: dict) -> Dict:
        data = self._packer.pack(payload)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self._send_and_recv(obs)

    def infer_realtime(
        self,
        obs: Dict,
        *,
        action_prefix: Optional[np.ndarray] = None,
        noise: Optional[np.ndarray] = None,
    ) -> Dict:
        """Inference with prefix conditioning for real-time chunking.

        Embeds action_prefix into the observation dict under a reserved key
        so the server can extract it and call policy.infer_realtime().
        """
        if action_prefix is None:
            return self.infer(obs)

        payload = dict(obs)
        payload[_ACTION_PREFIX_KEY] = np.asarray(action_prefix)
        return self._send_and_recv(payload)

    @override
    def reset(self) -> None:
        pass
