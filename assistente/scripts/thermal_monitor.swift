import Foundation
import IOKit

/// Monitor termico persistente: emette snapshot JSON su stdout a intervallo fisso.
///
/// Formato riga:
/// `{"thermalState":0,"tempC":54.0}` oppure `{"thermalState":1,"tempC":null}`
/// quando SMC non è disponibile. Lo stato Apple resta autorevole; la temperatura
/// è opzionale e usata solo per isteresi/anticipazione lato Python.
///
/// Uso:
///   swift thermal_monitor.swift [intervallo_ms]
/// Default intervallo: 2000 ms. Termina su SIGTERM/SIGINT o chiusura stdout.

let KERNEL_INDEX_SMC: UInt32 = 2
let chiaviTemperatura = ["Tp09", "Te05", "TC0P", "TC0D", "TC0E"]
let intervalloDefaultMs = 2000
let temperaturaMinimaPlausibile = 0.0
let temperaturaMassimaPlausibile = 150.0

/// Converte una stringa di 4 caratteri nel FourCC usato dalle chiavi SMC.
func fourCharCode(_ s: String) -> UInt32 {
    var risultato: UInt32 = 0
    // Ogni carattere occupa un byte del FourCC big-endian.
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

/// Decodifica il formato SP78 tipico dei sensori temperatura Apple SMC.
func decodeSp78(_ bytes: (UInt8, UInt8)) -> Double {
    Double(bytes.0) + Double(bytes.1) / 256.0
}

/// Legge una chiave temperatura SMC; nil se la connessione o il sensore falliscono.
func readTemp(conn: io_connect_t, keyName: String) -> Double? {
    var input = SMCParam()
    var output = SMCParam()
    input.key = fourCharCode(keyName)
    var inputSize = MemoryLayout<SMCParam>.size
    var outputSize = MemoryLayout<SMCParam>.size
    // IOConnectCallStructMethod fallisce su macchine senza quella chiave SMC.
    guard IOConnectCallStructMethod(
        conn, KERNEL_INDEX_SMC, &input, inputSize, &output, &outputSize
    ) == KERN_SUCCESS else { return nil }
    let temperatura = decodeSp78((output.bytes.bytes.0, output.bytes.bytes.1))
    // Scarta valori fuori range per non propagare letture sporche a Python.
    if temperatura > temperaturaMinimaPlausibile
        && temperatura < temperaturaMassimaPlausibile
    {
        return temperatura
    }
    return nil
}

/// Apre AppleSMC una sola volta; nil se IOKit non espone il servizio.
func apriConnessioneSMC() -> io_connect_t? {
    var conn: io_connect_t = 0
    let servizio = IOServiceGetMatchingService(
        kIOMainPortDefault,
        IOServiceMatching("AppleSMC")
    )
    // Su VM o hardware bloccato SMC può non esistere: lo snapshot resta senza temp.
    guard servizio != 0 else { return nil }
    defer { IOObjectRelease(servizio) }
    guard IOServiceOpen(servizio, mach_task_self_, 0, &conn) == KERN_SUCCESS else {
        return nil
    }
    return conn
}

/// Prova le chiavi note in ordine finché una restituisce una temperatura valida.
func leggiTemperatura(conn: io_connect_t?) -> Double? {
    guard let conn else { return nil }
    for nome in chiaviTemperatura {
        if let temperatura = readTemp(conn: conn, keyName: nome) {
            return temperatura
        }
    }
    return nil
}

/// Serializza lo snapshot in una sola riga JSON compatibile col parser Python.
func emettiSnapshot(stato: Int, temperatura: Double?) {
    let tempJson: String
    // null JSON esplicito evita ambiguità con 0.0 quando SMC è assente.
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
// `once` emette un solo snapshot (fallback Python) senza loop persistente.
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
    // Chiude IOKit solo se la connessione era stata aperta con successo.
    if let connessioneSMC {
        IOServiceClose(connessioneSMC)
    }
}

if oneShot {
    let stato = ProcessInfo.processInfo.thermalState.rawValue
    let temperatura = leggiTemperatura(conn: connessioneSMC)
    emettiSnapshot(stato: stato, temperatura: temperatura)
} else {
    // Loop persistente: un processo evita il costo di avvio Swift a ogni poll Python.
    while !deveTerminare {
        let stato = ProcessInfo.processInfo.thermalState.rawValue
        let temperatura = leggiTemperatura(conn: connessioneSMC)
        emettiSnapshot(stato: stato, temperatura: temperatura)
        // Sleep a slice per reagire rapidamente a SIGTERM senza busy-wait.
        let sliceMs = 50
        var atteso = 0
        while atteso < intervalloMs && !deveTerminare {
            usleep(useconds_t(sliceMs * 1000))
            atteso += sliceMs
        }
    }
}
