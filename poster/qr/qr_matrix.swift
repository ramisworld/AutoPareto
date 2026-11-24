// Print the QR module matrix (1 = dark) for argv[1], error correction argv[2], using macOS CoreImage only.
import Foundation
import CoreImage
import CoreGraphics

let msg = CommandLine.arguments[1]
let level = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "M"
let f = CIFilter(name: "CIQRCodeGenerator")!
f.setValue(msg.data(using: .utf8)!, forKey: "inputMessage")
f.setValue(level, forKey: "inputCorrectionLevel")
let img = f.outputImage!
let ctx = CIContext(options: nil)
let cg = ctx.createCGImage(img, from: img.extent)!
let w = cg.width, h = cg.height
var px = [UInt8](repeating: 0, count: w * h)
let gray = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                     space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
gray.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
for r in 0..<h { print(String((0..<w).map { px[r * w + $0] < 128 ? "1" : "0" })) }
