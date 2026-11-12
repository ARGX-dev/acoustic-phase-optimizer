// Plays a test signal and records the inputs in the same audio callback, so what was played (the reference) and
// what the mic heard line up sample for sample apart from the fixed latency of the audio chain.
// Messages in:  {type: 'start', signal: Float32Array, total: samples to record} · {type: 'stop'}
// Messages out: {type: 'level', peak: [per input channel], t: seconds} about 10 times a second
//               {type: 'done', inputs: [Float32Array per channel], ref: Float32Array}

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.channels = options.processorOptions?.channels || 1;
    this.running = false;
    this.finished = false;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'start') {
        this.signal = m.signal;
        this.total = m.total;
        this.inputs = Array.from({ length: this.channels }, () => new Float32Array(m.total));
        this.ref = new Float32Array(m.total);
        this.pos = 0;
        this.peak = new Array(this.channels).fill(0);
        this.nextLevel = 0;
        this.running = true;
      } else if (m.type === 'stop') {
        this.running = false;
        this.finished = true;
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const inp = inputs[0] || [];
    if (!this.running) {
      for (const c of out) c.fill(0);
      return !this.finished;
    }
    const n = out[0].length;
    const m = Math.min(n, this.total - this.pos);
    for (let i = 0; i < n; i++) {
      const k = this.pos + i;
      const v = i < m && k < this.signal.length ? this.signal[k] : 0;
      for (const c of out) c[i] = v;
      if (i < m) this.ref[k] = v;
    }
    for (let c = 0; c < this.channels; c++) {
      const src = inp[c];
      if (!src) continue;
      const dst = this.inputs[c];
      let p = this.peak[c];
      for (let i = 0; i < m; i++) {
        const x = src[i];
        dst[this.pos + i] = x;
        const a = x < 0 ? -x : x;
        if (a > p) p = a;
      }
      this.peak[c] = p;
    }
    this.pos += m;
    if (this.pos >= this.nextLevel) {
      this.port.postMessage({ type: 'level', peak: this.peak, t: this.pos / sampleRate, total: this.total / sampleRate });
      this.peak = new Array(this.channels).fill(0);
      this.nextLevel = this.pos + Math.round(sampleRate / 10);
    }
    if (this.pos >= this.total) {
      this.running = false;
      this.finished = true;
      this.port.postMessage({ type: 'done', inputs: this.inputs, ref: this.ref }, [...this.inputs.map((x) => x.buffer), this.ref.buffer]);
      return false;
    }
    return true;
  }
}

registerProcessor('apo-capture', CaptureProcessor);
