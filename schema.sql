create table relation (
    type char(16),
    property char(16),
    unit char(16),
    primary key (type, property, unit)
);

create table aliases (
    wid char(16),
    alias char(50),
    primary key (wid, alias)
);


create table distribution (
    type char(16),
    property char(16),
    unit char(16),
    amount REAL,
    primary key (type, property, unit, amount),
    foreign key (type) references relation(type),
    foreign key (property) references relation(property),
    foreign key (unit) references relation(unit)
);

CREATE TABLE "mapping" (
    wid char(16),
    label char(100) collate nocase,
    primary key (wid)
);

CREATE TABLE "type" (
    wid char(16),
    types char(512),
    primary key (wid)
);

create table means (
    type char(16),
    property char(16),
    unit char(16),
    mean REAL,
    std REAL,
    plabel char(64),
    primary key (type, property, unit),
    foreign key (type) references relation(type),
    foreign key (property) references relation(property),
    foreign key (unit) references relation(unit)
);

-- create index typeInd on relation(type);
-- create index propertyInd on relation(property);
-- create index unitInd on relation(unit);

create index aliasInd on aliases(canonical);
create index labelInd on mapping(label);

