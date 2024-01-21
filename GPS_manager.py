# FIXME -> DELETE THIS
from random import random
from threading import Thread


class GPSManager:
    def __init__(self):
        self._lat = None
        self._long = None
        self._is_running = True

        thread = Thread(target=self._update_coords)
        thread.start()

    def _update_coords(self):
        while self._is_running:
            self._lat = random() + .1
            self._long = random() + .1

    def get_current_coords(self):
        return {'lat': self._lat, 'long': self._long}

    def stop_updating_coords(self):
        self._is_running = False
