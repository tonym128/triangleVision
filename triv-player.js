/**
 * TrivPlayer - A high-performance JavaScript library for playback of .triv files.
 * Uses Canvas 2D for rendering and fflate for zlib decompression.
 */
class TrivPlayer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false });
        this.options = {
            loop: options.loop || false,
            onLoad: options.onLoad || (() => {}),
            onFrame: options.onFrame || (() => {}),
            onEnd: options.onEnd || (() => {})
        };

        this.header = null;
        this.frames = [];
        this.currentFrame = 0;
        this.isPlaying = false;
        this.timer = null;
        this.dataView = null;
        this.offset = 0;
    }

    /**
     * Load a .triv file from a URL or ArrayBuffer.
     */
    async load(source) {
        let buffer;
        if (typeof source === 'string') {
            const response = await fetch(source);
            buffer = await response.arrayBuffer();
        } else {
            buffer = source;
        }

        this.dataView = new DataView(buffer);
        this.buffer = new Uint8Array(buffer);
        this.offset = 0;

        // Parse Header (10 bytes)
        const magic = String.fromCharCode(...this.buffer.slice(0, 4));
        if (magic !== 'TRIV') {
            throw new Error('Not a valid .triv file');
        }

        this.header = {
            width: this.dataView.getUint16(4, true),
            height: this.dataView.getUint16(6, true),
            fps: this.dataView.getUint16(8, true)
        };

        this.canvas.width = this.header.width;
        this.canvas.height = this.header.height;
        this.offset = 10;

        // Pre-parse frame offsets for seeking and smooth playback
        this.frameOffsets = [];
        while (this.offset < this.buffer.length) {
            const type = String.fromCharCode(this.buffer[this.offset]);
            const length = this.dataView.getUint32(this.offset + 1, true);
            this.frameOffsets.push({
                offset: this.offset + 5,
                length: length
            });
            this.offset += 5 + length;
        }

        this.options.onLoad(this.header);
        this.renderFrame(0);
    }

    /**
     * Decompress and render a specific frame.
     */
    async renderFrame(index) {
        if (index < 0 || index >= this.frameOffsets.length) return;
        
        const frameInfo = this.frameOffsets[index];
        const compressed = this.buffer.slice(frameInfo.offset, frameInfo.offset + frameInfo.length);
        
        if (typeof fflate === 'undefined' || !fflate.unzlibSync) {
            throw new Error('fflate library (unzlibSync) is required for decompression');
        }

        const decompressed = fflate.unzlibSync(compressed);
        // Use the byteOffset and byteLength explicitly to create a scoped DataView
        const view = new DataView(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength);
        
        let ptr = 0;
        const numPoints = view.getUint16(ptr, true);
        const numSimplices = view.getUint16(ptr + 2, true);
        const numColors = view.getUint16(ptr + 4, true);
        ptr += 6;

        // Use a loop to avoid alignment issues with TypedArray constructors on shared buffers
        const points = new Uint16Array(numPoints * 2);
        for (let i = 0; i < numPoints * 2; i++) {
            points[i] = view.getUint16(ptr, true);
            ptr += 2;
        }

        const simplices = new Uint16Array(numSimplices * 3);
        for (let i = 0; i < numSimplices * 3; i++) {
            simplices[i] = view.getUint16(ptr, true);
            ptr += 2;
        }

        const colors = new Uint8Array(decompressed.buffer, decompressed.byteOffset + ptr, numColors * 3);

        // Draw to Canvas
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        for (let i = 0; i < numSimplices; i++) {
            const i1 = simplices[i * 3];
            const i2 = simplices[i * 3 + 1];
            const i3 = simplices[i * 3 + 2];

            // Colors are BGR (3 bytes per triangle)
            const b = colors[i * 3];
            const g = colors[i * 3 + 1];
            const r = colors[i * 3 + 2];

            this.ctx.fillStyle = `rgb(${r},${g},${b})`;
            this.ctx.beginPath();
            this.ctx.moveTo(points[i1 * 2], points[i1 * 2 + 1]);
            this.ctx.lineTo(points[i2 * 2], points[i2 * 2 + 1]);
            this.ctx.lineTo(points[i3 * 2], points[i3 * 2 + 1]);
            this.ctx.closePath();
            this.ctx.fill();
        }

        this.currentFrame = index;
        this.options.onFrame(index, this.frameOffsets.length);
    }

    play() {
        if (this.isPlaying) return;
        this.isPlaying = true;
        
        const delay = 1000 / this.header.fps;
        const next = async () => {
            if (!this.isPlaying) return;
            
            await this.renderFrame(this.currentFrame);
            this.currentFrame++;
            
            if (this.currentFrame >= this.frameOffsets.length) {
                if (this.options.loop) {
                    this.currentFrame = 0;
                } else {
                    this.isPlaying = false;
                    this.options.onEnd();
                    return;
                }
            }
            
            this.timer = setTimeout(next, delay);
        };
        
        next();
    }

    pause() {
        this.isPlaying = false;
        if (this.timer) clearTimeout(this.timer);
    }

    seek(percent) {
        const index = Math.floor(percent * (this.frameOffsets.length - 1));
        this.renderFrame(index);
    }
}
