import Foundation
import IOKit

/// Persistent thermal monitor: emits JSON snapshots on stdout at a fixed interval.
///
/// Line format:
/// `{"thermalState":0,"tempC":54.0}` or `{"thermalState":1,"tempC":null}`
/// when the SMC is not available. The Apple state stays authoritative; the temperature
/// is optional and used only for hysteresis/lookahead on the Python side.
///
/// Usage:
///   swift thermal_monitor.swift [interval_ms]
/// Default interval: 2000 ms. Exits on SIGTERM/SIGINT or when stdout closes.

let KERNEL_INDEX_SMC: UInt32 = 2
let chiaviTemperatura = ["Tp09", "Te05", "TC0P", "TC0D", "TC0E"]
let intervalloDefaultMs = 2000
let temperaturaMinimaPlausibile = 0.0
let temperaturaMassimaPlausibile = 150.0

/// Convert a 4-character string into the FourCC used by SMC keys.
func fourCharCode(_ s: String) -> UInt32 {
    var risultato: UInt32 = 0
    // Each character takes one byte of the big-endian FourCC.
    for carattere in s.utf16 {
        risultato = (risultato << 8) + UInt32(carattere)
    }
    return risultato
}

struct SMCBytes {
    var bytes: (
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8
    ) = (
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    )
}

struct SMCParam {
    var key: UInt32 = 0
    var vers = SMCBytes()
    var pLimitData = SMCBytes()
    var keyInfo = SMCBytes()
    var result: UInt8 = 0
    var status: UInt8 = 0
    var data8: UInt8 = 0
    var data32: UInt32 = 0
    var bytes = SMCBytes()
}

/// Decode the SP78 format typical of Apple SMC temperature sensors.
func decodeSp78(_ bytes: (UInt8, UInt8)) -> Double {
    Double(bytes.0) + Double(bytes.1) / 256.0
}

/// Read an SMC temperature key; nil if the connection or the sensor fails.
func readTemp(conn: io_connect_t, keyName: String) -> Double? {
    var input = SMCParam()
    var output = SMCParam()
    input.key = fourCharCode(keyName)
    var inputSize = MemoryLayout<SMCParam>.size
    var outputSize = MemoryLayout<SMCParam>.size
    // IOConnectCallStructMethod fails on machines without that SMC key.
    guard IOConnectCallStructMethod(
        conn, KERNEL_INDEX_SMC, &input, inputSize, &output, &outputSize
    ) == KERN_SUCCESS else { return nil }
    let temperatura = decodeSp78((output.bytes.bytes.0, output.bytes.bytes.1))
    // Drop out-of-range values so dirty readings do not reach Python.
    if temperatura > temperaturaMinimaPlausibile
        && temperatura < temperaturaMassimaPlausibile
    {
        return temperatura
    }
    return nil
}

/// Open AppleSMC only once; nil if IOKit does not expose the service.
func apriConnessioneSMC() -> io_connect_t? {
    var conn: io_connect_t = 0
    let servizio = IOServiceGetMatchingService(
        kIOMainPortDefault,
        IOServiceMatching("AppleSMC")
    )
    // On a VM or locked-down hardware the SMC may not exist: the snapshot stays without a temperature.
    guard servizio != 0 else { return nil }
    defer { IOObjectRelease(servizio) }
    guard IOServiceOpen(servizio, mach_task_self_, 0, &conn) == KERN_SUCCESS else {
        return nil
    }
    return conn
}

/// Try the known keys in order until one returns a valid temperature.
func leggiTemperatura(conn: io_connect_t?) -> Double? {
    guard let conn else { return nil }
    for nome in chiaviTemperatura {
        if let temperatura = readTemp(conn: conn, keyName: nome) {
            return temperatura
        }
    }
    return nil
}

/// Serialize the snapshot into a single JSON line compatible with the Python parser.
func emettiSnapshot(stato: Int, temperatura: Double?) {
    let tempJson: String
    // An explicit JSON null avoids ambiguity with 0.0 when the SMC is missing.
    if let temperatura {
        tempJson = String(format: "%.1f", temperatura)
    } else {
        tempJson = "null"
    }
    print("{\"thermalState\":\(stato),\"tempC\":\(tempJson)}")
    fflush(stdout)
}

let argomenti = CommandLine.arguments
var intervalloMs = intervalloDefaultMs
var oneShot = false
// `once` emits a single snapshot (Python fallback) without the persistent loop.
for argomento in argomenti.dropFirst() {
    if argomento == "once" || argomento == "--once" {
        oneShot = true
        continue
    }
    if let valore = Int(argomento), valore > 0 {
        intervalloMs = valore
    }
}

var deveTerminare = false
signal(SIGTERM) { _ in deveTerminare = true }
signal(SIGINT) { _ in deveTerminare = true }

let connessioneSMC = apriConnessioneSMC()
defer {
    // Close IOKit only if the connection was opened successfully.
    if let connessioneSMC {
        IOServiceClose(connessioneSMC)
    }
}

if oneShot {
    let stato = ProcessInfo.processInfo.thermalState.rawValue
    let temperatura = leggiTemperatura(conn: connessioneSMC)
    emettiSnapshot(stato: stato, temperatura: temperatura)
} else {
    // Persistent loop: one process avoids the Swift startup cost on every Python poll.
    while !deveTerminare {
        let stato = ProcessInfo.processInfo.thermalState.rawValue
        let temperatura = leggiTemperatura(conn: connessioneSMC)
        emettiSnapshot(stato: stato, temperatura: temperatura)
        // Sleep in slices to react quickly to SIGTERM without busy-waiting.
        let sliceMs = 50
        var atteso = 0
        while atteso < intervalloMs && !deveTerminare {
            usleep(useconds_t(sliceMs * 1000))
            atteso += sliceMs
        }
    }
}
