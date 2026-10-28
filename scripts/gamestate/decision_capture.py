"""Bounded asynchronous capture of exact policy tensors, never a control input."""
import json
from pathlib import Path
from queue import Queue, Full
from threading import Thread
import numpy as np


class DecisionCapture:
    def __init__(self, output, *, every=10, limit=2000, capacity=4):
        if every < 1 or not 1 <= limit <= 10000 or capacity < 1:
            raise ValueError('Positive sampling interval and bounded capture required')
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.every=every;self.limit=limit;self.seen=0;self.accepted=0
        self.dropped=0;self.written=0;self.error=None;self.closed=False
        self.queue=Queue(maxsize=capacity)
        self.worker=Thread(target=self._write,daemon=True);self.worker.start()

    def capture(self, planes, scalars, mask, decision, context):
        self.seen+=1
        if self.closed or self.error or self.accepted>=self.limit or (self.seen-1)%self.every:
            return
        # Own the arrays before the next frame mutates hand, mask or observation.
        item=(self.seen,np.array(planes,copy=True),np.array(scalars,copy=True),
              np.array(mask,copy=True),json.dumps(dict(context,decision=decision)))
        try:self.queue.put_nowait(item)
        except Full:self.dropped+=1
        else:self.accepted+=1

    def _write(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                if self.error:continue
                index,planes,scalars,mask,metadata=item
                with (self.output/f'{index:06d}.npz').open('xb') as stream:
                    np.savez_compressed(stream,planes=planes,scalars=scalars,
                                        action_mask=mask,metadata=np.asarray(metadata))
                self.written+=1
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}'
            finally:self.queue.task_done()

    def close(self):
        if self.closed:return
        self.closed=True;self.queue.put(None);self.worker.join()
        (self.output/'summary.json').write_text(json.dumps(dict(
            seen=self.seen,accepted=self.accepted,written=self.written,dropped=self.dropped,
            error=self.error,every=self.every,limit=self.limit),indent=2)+'\n')
