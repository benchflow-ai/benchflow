//! ELF load references, as facts (TR-PROTOCOL.md §3.4): PT_INTERP, and from
//! PT_DYNAMIC the DT_NEEDED, DT_RPATH, DT_RUNPATH, DT_AUDIT, DT_DEPAUDIT,
//! DT_FILTER and DT_AUXILIARY strings (DT_STRTAB is mapped to a file offset
//! through the PT_LOAD segments), DT_FLAGS and DT_FLAGS_1, and every tag
//! outside [`KNOWN_TAGS`] as `unknown_tags`. Only the parts named by the
//! headers are read, with positioned reads, so a large library costs a few
//! reads. The structures are decoded by the `object` crate.

use object::elf::{self, FileHeader32, FileHeader64};
use object::read::elf::{Dyn, FileHeader, ProgramHeader};
use object::{pod, Endianness};
use rustix::fd::BorrowedFd;
use serde_json::{json, Value};
use std::mem::size_of;

const LIMIT: u64 = 16 << 20;

fn pread(fd: BorrowedFd, off: u64, len: u64) -> Option<Vec<u8>> {
    if len > LIMIT {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    let mut got = 0usize;
    while got < buf.len() {
        match rustix::io::pread(fd, &mut buf[got..], off + got as u64) {
            Ok(0) => return None,
            Ok(n) => got += n,
            Err(rustix::io::Errno::INTR) => {}
            Err(_) => return None,
        }
    }
    Some(buf)
}

fn string_at(tab: &[u8], off: u64) -> Option<String> {
    let s = tab.get(off as usize..)?;
    let end = s.iter().position(|b| *b == 0)?;
    Some(String::from_utf8_lossy(&s[..end]).into_owned())
}

/// Dynamic tags that only describe the object's own tables, relocation,
/// versioning, initialisers or hashing: they name no file and change no
/// search. The string tags BenchFlow reports by name (DT_NEEDED, DT_RPATH,
/// DT_RUNPATH, DT_AUDIT, DT_DEPAUDIT, DT_FILTER, DT_AUXILIARY) and the flag
/// words are reported in their own fields. Any other tag is `unknown_tags`.
pub const KNOWN_TAGS: &[i64] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    32, 33, 34, 35, 36, 37, // through DT_RELRENT; 31 is DT_ENCODING's unused slot
    0x6ffffdf5, 0x6ffffdf6, 0x6ffffdf7, 0x6ffffdf8, 0x6ffffdf9, 0x6ffffdfa, 0x6ffffdfb, 0x6ffffdfc, 0x6ffffdfd, 0x6ffffdfe, 0x6ffffdff, // GNU_PRELINKED..SYMINENT
    0x6ffffef5, 0x6ffffef6, 0x6ffffef7, 0x6ffffef8, 0x6ffffef9, 0x6ffffefa, 0x6ffffeff, // GNU_HASH, TLSDESC_PLT/GOT, GNU_CONFLICT, GNU_LIBLIST, CONFIG, SYMINFO
    0x6ffffff0, 0x6ffffff9, 0x6ffffffa, 0x6ffffffb, 0x6ffffffc, 0x6ffffffd, 0x6ffffffe, 0x6fffffff, // VERSYM, RELACOUNT, RELCOUNT, FLAGS_1, VERDEF(NUM), VERNEED(NUM)
    0x70000001, 0x70000003, 0x70000005, // AARCH64_BTI_PLT, AARCH64_PAC_PLT, AARCH64_VARIANT_PCS
];

fn facts_for<H: FileHeader<Endian = Endianness>>(fd: BorrowedFd, class: u32) -> Result<Value, &'static str> {
    let head = pread(fd, 0, size_of::<H>() as u64).ok_or("short header")?;
    let h = H::parse(&head[..]).map_err(|_| "bad header")?;
    let e = h.endian().map_err(|_| "bad endianness")?;
    let phoff: u64 = h.e_phoff(e).into();
    let phnum = h.e_phnum(e) as usize;
    let entsize = h.e_phentsize(e) as usize;
    let mut interp = None;
    let mut loads = Vec::new();
    let mut dynamic = None;
    if phoff != 0 && phnum != 0 {
        if entsize != size_of::<H::ProgramHeader>() || phnum == elf::PN_XNUM as usize {
            return Err("unusual program headers");
        }
        let buf = pread(fd, phoff, (phnum * entsize) as u64).ok_or("short program headers")?;
        let (phdrs, _) = pod::slice_from_bytes::<H::ProgramHeader>(&buf, phnum).map_err(|_| "bad program headers")?;
        for ph in phdrs {
            let (off, size): (u64, u64) = (ph.p_offset(e).into(), ph.p_filesz(e).into());
            let t = ph.p_type(e);
            if t == elf::PT_INTERP {
                let d = pread(fd, off, size).ok_or("short PT_INTERP")?;
                let end = d.iter().position(|b| *b == 0).unwrap_or(d.len());
                interp = Some(String::from_utf8_lossy(&d[..end]).into_owned());
            } else if t == elf::PT_LOAD {
                loads.push((ph.p_vaddr(e).into(), off, size));
            } else if t == elf::PT_DYNAMIC {
                dynamic = Some((off, size));
            }
        }
    }
    let mut strings: [Vec<String>; 7] = Default::default();
    let (mut flags, mut flags_1) = (0u64, 0u64);
    let mut unknown: Vec<i64> = Vec::new();
    if let Some((off, size)) = dynamic {
        let buf = pread(fd, off, size).ok_or("short PT_DYNAMIC")?;
        let n = buf.len() / size_of::<H::Dyn>();
        let (dyns, _) = pod::slice_from_bytes::<H::Dyn>(&buf, n).map_err(|_| "bad PT_DYNAMIC")?;
        let (mut strtab, mut strsz) = (None, 0u64);
        let mut refs = Vec::new();
        for d in dyns {
            let tag = d.d_tag(e);
            let val: u64 = d.d_val(e).into();
            if tag == elf::DT_NULL {
                break;
            } else if tag == elf::DT_STRTAB {
                strtab = Some(val);
            } else if tag == elf::DT_STRSZ {
                strsz = val;
            } else if tag == elf::DT_FLAGS {
                flags = val;
            } else if tag == elf::DT_FLAGS_1 {
                flags_1 = val;
            } else if let Some(i) = STRING_TAGS.iter().position(|t| *t == tag) {
                refs.push((i, val));
            } else if !KNOWN_TAGS.contains(&tag.0) && !unknown.contains(&tag.0) {
                unknown.push(tag.0);
            }
        }
        if !refs.is_empty() {
            let vaddr = strtab.ok_or("DT_STRTAB missing")?;
            let off = loads
                .iter()
                .find(|(va, _, sz)| vaddr >= *va && vaddr < *va + *sz)
                .map(|(va, o, _)| vaddr - va + o)
                .ok_or("DT_STRTAB outside every PT_LOAD")?;
            let tab = pread(fd, off, strsz).ok_or("short DT_STRTAB")?;
            for (i, val) in refs {
                let s = string_at(&tab, val).ok_or("a dynamic string is outside DT_STRTAB")?;
                strings[i].push(s);
            }
        }
    }
    let [needed, rpath, runpath, audit, depaudit, filter, auxiliary] = strings;
    Ok(json!({
        "class": class,
        "machine": h.e_machine(e).0,
        "elf_type": h.e_type(e).0,
        "interp": interp,
        "dynamic": dynamic.is_some(),
        "static": interp.is_none() && needed.is_empty(),
        "needed": needed,
        "rpath": rpath,
        "runpath": runpath,
        "audit": audit,
        "depaudit": depaudit,
        "filter": filter,
        "auxiliary": auxiliary,
        "flags": flags,
        "flags_1": flags_1,
        "pie": flags_1 & elf::DF_1_PIE.0 as u64 != 0,
        "unknown_tags": unknown,
    }))
}

/// The string tags reported by name, in the order of the fields.
const STRING_TAGS: [elf::DynamicTag; 7] = [
    elf::DT_NEEDED,
    elf::DT_RPATH,
    elf::DT_RUNPATH,
    elf::DT_AUDIT,
    elf::DT_DEPAUDIT,
    elf::DT_FILTER,
    elf::DT_AUXILIARY,
];

/// ELF facts of an open regular file, None when it is not ELF.
pub fn facts(fd: BorrowedFd, magic: &[u8]) -> Option<Value> {
    if magic.len() < 5 || &magic[..4] != b"\x7fELF" {
        return None;
    }
    let r = match magic[4] {
        1 => facts_for::<FileHeader32<Endianness>>(fd, 32),
        2 => facts_for::<FileHeader64<Endianness>>(fd, 64),
        _ => Err("unknown ELF class"),
    };
    Some(r.unwrap_or_else(|why| json!({"error": why})))
}
