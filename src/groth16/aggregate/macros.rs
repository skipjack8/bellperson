/// returns an Fr element derived from sha256 hash of all elements that gets
/// passed to the macro. It runs in a loop in case the Fr element doesn't have
/// an inverse and prepends a monotically increasing counter before
/// each hash input.
macro_rules! oracle {
    // https://fromherotozero.dev/blog/introduction-to-rust-macros/
    ( $( $x:expr), * ) => { {
        let mut counter_nonce: usize = 0;
        let one = E::Fr::one();
        let r = loop {
            counter_nonce += 1;
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            $(
                bincode::serialize_into(&mut hash_input, $x).expect("vec");
            )*
            let d = &Sha256::digest(&hash_input);
            if let Some(c) = E::Fr::from_random_bytes(&d) {
                if c == one {
                    continue;
                }
                if let Some(_) = c.inverse() {
                    break c;
                }
            }
        };
        r
    }};
}

macro_rules! try_par {
    ($(let $name:ident = $f:expr),+) => {
        $(
            let mut $name = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name = &mut $name;
                    s.spawn(move |_| {
                        *$name = Some($f);
                    });)+
            });
        $(
            let $name = $name.unwrap()?;
        )+
    };
}

macro_rules! par {
    ($(let $name:ident = $f:expr),+) => {
        $(
            let mut $name = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name = &mut $name;
                    s.spawn(move |_| {
                        *$name = Some($f);
                    });)+
            });
        $(
            let $name = $name.unwrap();
        )+
    };

    ($(let ($name1:ident, $name2:ident) = $f:block),+) => {
        $(
            let mut $name1 = None;
            let mut $name2 = None;
        )+
            rayon::scope(|s| {
                $(
                    let $name1 = &mut $name1;
                    let $name2 = &mut $name2;
                    s.spawn(move |_| {
                        let (a, b) = $f;
                        *$name1 = Some(a);
                        *$name2 = Some(b);
                    });)+
            });
        $(
            let $name1 = $name1.unwrap();
            let $name2 = $name2.unwrap();
        )+
    }
}

macro_rules! mul {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.mul_assign($b);
        a
    }};
}

macro_rules! add {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.add_assign($b);
        a
    }};
}

macro_rules! sub {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.sub_assign($b);
        a
    }};
}
