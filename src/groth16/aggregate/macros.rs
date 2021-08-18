macro_rules! try_par {
    ($(let $name:ident = $f:expr),+) => {
        $(
            let mut $name = None;
        )+
            crate::multicore::THREAD_POOL.scoped(|s| {
                $(
                    let $name = &mut $name;
                    s.execute(move || {
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
            crate::multicore::THREAD_POOL.scoped(|s| {
                $(
                    let $name = &mut $name;
                    s.execute(move || {
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
            crate::multicore::THREAD_POOL.scoped(|s| {
                $(
                    let $name1 = &mut $name1;
                    let $name2 = &mut $name2;
                    s.execute(move || {
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
