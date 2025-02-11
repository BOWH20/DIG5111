NoteC = note(0.7,40,2);
NoteCS = note(0.7,41,2);
NoteD = note(0.7,42,2);
NoteDS = note(0.7,43,2);
NoteE = note(0.7,44,2);
NoteES = note(0.7,45,2);
NoteF = note(0.7,46,2);
NoteFsharp = note(0.7,47,2);
NoteG = note(0.7,48,2);
NoteGS = note(0.7,49,2);
NoteA = note(0.7,50,2);
NoteAS = note(0.7,51,2);
NoteB = note(0.7,52,2);
NoteBS = note(0.7,53,2);
NoteRest = note(0,44,0.5);

NoteCshort = note(0.7,40,0.5);
NoteCSshort = note(0.7,41,0.5);
NoteDshort = note(0.7,42,0.5);
NoteDSshort = note(0.7,43,0.5);
NoteEshort = note(0.7,44,0.5);
NoteESshort = note(0.7,45,0.5);
NoteFshort = note(0.7,46,0.5);
NoteFsharpshort = note(0.7,47,0.5);
NoteGshort = note(0.7,48,0.5);
NoteGSshort = note(0.7,49,0.5);
NoteAshort = note(0.7,50,0.5);
NoteASshort = note(0.7,51,0.5);
NoteBshort = note(0.7,52,0.5);
NoteBSshort = note(0.7,53,0.5);

lead_org = NoteAshort;
lead = lead_org./max(abs(lead_org));

lead2_org = NoteBSshort;
lead2 = lead2_org./max(abs(lead2_org));

lead3_org = NoteFshort;
lead3 = lead3_org./max(abs(lead3_org));

A_org = NoteA+NoteC+NoteE+NoteFsharp;
achord = A_org./max(abs(A_org));

Ctriad_org = NoteC+NoteE+NoteG;
Ctriad = Ctriad_org./max(abs(Ctriad_org));

NoteRest_org = NoteRest;
rest = NoteRest_org./max(abs(NoteRest_org));


a = [Ctriad achord ];

l = [lead lead2 rest lead3];

sound(a)
sound(l)