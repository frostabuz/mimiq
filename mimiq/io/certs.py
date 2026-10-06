"""Local HTTPS certificates for Mimiq Link (Safari only grants camera access on HTTPS).

A private "Mimiq Local CA" signs a short-lived server certificate for the PC's LAN IPs.
Users can simply accept Safari's warning once, or install the CA on the iPhone
(Settings → General → About → Certificate Trust Settings) for a warning-free link.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import json
import logging
import re
import socket
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("mimiq.certs")


def local_ipv4s() -> List[str]:
    """LAN IPv4 addresses, the one on the default route first."""
    found: List[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except Exception:  # OSError, or UnicodeError for non-ASCII computer names
        pass
    out: List[str] = []
    for ip in found:
        if ip.startswith("127.") or ip.startswith("169.254.") or ip in out:
            continue
        out.append(ip)

    def rank(ip: str) -> int:
        if ip == (found[0] if found else ""):
            return 0
        if ip.startswith("192.168."):
            return 1
        if ip.startswith("10."):
            return 2
        return 3
    return sorted(out, key=rank)


# The local CA may only vouch for private-network addresses: even if its key leaked it could not be
# used to impersonate public websites on the iPhone (X.509 name constraints, RFC 5280 §4.2.1.10).
PRIVATE_NETS = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16", "127.0.0.0/8"]
PRIVATE_DNS = ["local", "localhost"]


def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in ipaddress.ip_network(n) for n in PRIVATE_NETS)


def _dns_label(host: str) -> str:
    """Computer name as a valid DNS label (Windows names may contain Cyrillic or spaces)."""
    label = re.sub(r"[^A-Za-z0-9-]", "", host or "")[:63].strip("-")
    return label.lower()


def _name(cn: str):
    from cryptography.x509.oid import NameOID
    from cryptography import x509
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Mimiq")])


def ensure_certificates(folder: Path, ips: List[str]) -> Tuple[Path, Path, Path]:
    """Return (fullchain.pem, server.key, ca.crt); (re)issue the server cert when IPs change."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID

    folder.mkdir(parents=True, exist_ok=True)
    ca_key_p, ca_crt_p = folder / "ca.key", folder / "ca.crt"
    key_p, chain_p, meta_p = folder / "server.key", folder / "fullchain.pem", folder / "server.json"
    now = dt.datetime.now(dt.timezone.utc)

    if ca_key_p.exists() and ca_crt_p.exists():
        ca_key = serialization.load_pem_private_key(ca_key_p.read_bytes(), None)
        ca_crt = x509.load_pem_x509_certificate(ca_crt_p.read_bytes())
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_crt = (x509.CertificateBuilder()
                  .subject_name(_name("Mimiq Local CA")).issuer_name(_name("Mimiq Local CA"))
                  .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
                  .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=3650))
                  .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                  .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                               content_commitment=False, key_encipherment=False,
                                               data_encipherment=False, key_agreement=False,
                                               encipher_only=False, decipher_only=False), critical=True)
                  .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
                  .add_extension(x509.NameConstraints(
                      permitted_subtrees=[x509.IPAddress(ipaddress.ip_network(n)) for n in PRIVATE_NETS]
                      + [x509.DNSName(d) for d in PRIVATE_DNS],
                      excluded_subtrees=None), critical=True)
                  .sign(ca_key, hashes.SHA256()))
        ca_key_p.write_bytes(ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
        ca_crt_p.write_bytes(ca_crt.public_bytes(serialization.Encoding.PEM))

    try:
        host = _dns_label(socket.gethostname())
    except Exception:
        host = ""
    wanted = {"ips": sorted({ip for ip in ips + ["127.0.0.1"] if _is_private(ip)}), "host": host}
    if chain_p.exists() and key_p.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            leaf = x509.load_pem_x509_certificate(chain_p.read_bytes())
            not_after = leaf.not_valid_after_utc if hasattr(leaf, "not_valid_after_utc") else leaf.not_valid_after.replace(tzinfo=dt.timezone.utc)
            if meta == wanted and not_after - now > dt.timedelta(days=20):
                return chain_p, key_p, ca_crt_p
        except Exception:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    san = [x509.DNSName("localhost"), x509.DNSName("mimiq.local")]
    if host and host != "mimiq":
        san.append(x509.DNSName(f"{host}.local"))
    san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in wanted["ips"]]
    leaf = (x509.CertificateBuilder()
            .subject_name(_name(f"Mimiq Link ({host or 'PC'})")).issuer_name(ca_crt.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=True, content_commitment=False,
                                         data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                        serialization.NoEncryption()))
    chain_p.write_bytes(leaf.public_bytes(serialization.Encoding.PEM) + ca_crt.public_bytes(serialization.Encoding.PEM))
    meta_p.write_text(json.dumps(wanted))
    log.info("issued Mimiq Link certificate for %s", ", ".join(wanted["ips"]))
    return chain_p, key_p, ca_crt_p
