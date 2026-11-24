// Decode every QR code in the image argv[1] with CoreImage's CIDetector; print each message on its own line.
import Foundation
import CoreImage
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let img = CIImage(contentsOf: url) else { print("ERROR: cannot read image"); exit(2) }
let det = CIDetector(ofType: CIDetectorTypeQRCode, context: nil, options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])!
let found = det.features(in: img).compactMap { ($0 as? CIQRCodeFeature)?.messageString }
if found.isEmpty { print("NONE") }
for m in found { print(m) }
