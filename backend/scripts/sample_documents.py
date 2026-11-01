"""Twenty fictional documents for demos and for checking that the tools run.

Every one is invented, for a college that does not exist. They are not evidence of how
CampusMind performs on real documents.
"""

from dataclasses import dataclass, field

BANNER = "Sample College of Engineering. Demonstration document with invented content."


@dataclass(frozen=True)
class Sample:
    """A document, and questions its text can answer."""

    filename: str
    title: str
    doc_type: str
    body: str
    department: str | None = None
    semester: int | None = None
    questions: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        """Return the full document."""
        return f"{self.title}\n\n{BANNER}\n\n{self.body}\n"


SAMPLES: tuple[Sample, ...] = (
    Sample(
        "regulations_attendance.txt",
        "Attendance Regulations",
        "regulation",
        "Every student must have at least 75 percent attendance in each course to sit the end "
        "semester examination.\n\nThe Dean of Academics may condone attendance down to 65 "
        "percent on medical grounds, with a certificate submitted within seven days of "
        "returning.\n\nStudents below 65 percent must repeat the course.",
        questions=(
            ("What attendance do I need to sit the end semester exam?", "75 percent"),
            ("How low can attendance be condoned on medical grounds?", "65 percent"),
        ),
    ),
    Sample(
        "regulations_grading.txt",
        "Grading Regulations",
        "regulation",
        "Grades are awarded on a ten point scale. A grade of S carries 10 points and requires "
        "90 marks or more.\n\nA grade of F carries no points and means the course must be "
        "repeated.\n\nThe pass mark is 40 percent in the end semester examination and 50 "
        "percent overall.",
        questions=(
            ("What marks are needed for an S grade?", "90 marks or more"),
            ("What is the overall pass mark?", "50 percent"),
        ),
    ),
    Sample(
        "regulations_conduct.txt",
        "Examination Conduct Rules",
        "regulation",
        "Students must carry their identity card and hall ticket to every examination.\n\n"
        "Mobile phones are not allowed in the examination hall. Students may not enter more "
        "than 30 minutes after the start, or leave within the first hour.",
        questions=(("How late can I enter the exam hall?", "30 minutes after the start"),),
    ),
    Sample(
        "regulations_hostel.txt",
        "Hostel Rules",
        "regulation",
        "Hostel gates close at 10 pm on weekdays and 11 pm on weekends.\n\nVisitors are allowed "
        "in the common room between 4 pm and 7 pm. Cooking in rooms is not permitted.",
        questions=(("When do hostel gates close on weekdays?", "10 pm"),),
    ),
    Sample(
        "circular_exam_fee_2026.txt",
        "Circular on Examination Fees 2026-27",
        "circular",
        "The examination fee for the 2026-27 session is 1200 rupees per semester.\n\nFees must "
        "be paid by 30 October. Late payment attracts a fine of 100 rupees per day.",
        questions=(
            ("What is the examination fee?", "1200 rupees per semester"),
            ("What is the fine for paying the exam fee late?", "100 rupees per day"),
        ),
    ),
    Sample(
        "circular_scholarship.txt",
        "Circular on Merit Scholarships",
        "circular",
        "Merit scholarships are open to students with a CGPA of 8.5 or above.\n\nApplications "
        "close on 15 August and must be submitted to the Office of Student Affairs.",
        questions=(("What CGPA do I need for a merit scholarship?", "8.5 or above"),),
    ),
    Sample(
        "circular_library.txt",
        "Circular on Library Timings",
        "circular",
        "From 1 July the central library opens from 8 am to 9 pm on weekdays and 10 am to 5 pm "
        "on Saturdays. It is closed on Sundays.\n\nBooks may be borrowed for 14 days. The fine "
        "is 5 rupees per day after that.",
        questions=(
            ("How long can I keep a library book?", "14 days"),
            ("What are the library timings on Saturday?", "10 am to 5 pm"),
        ),
    ),
    Sample(
        "notice_hall_tickets.txt",
        "Notice on Hall Tickets",
        "notice",
        "Hall tickets for the end semester examinations will be issued from 20 November at "
        "the Examination Cell, between 10 am and 4 pm.\n\nStudents with fee dues will not be "
        "issued a hall ticket.",
        questions=(("When will hall tickets be issued?", "from 20 November"),),
    ),
    Sample(
        "notice_placement_drive.txt",
        "Notice on the Placement Drive",
        "notice",
        "The campus placement drive begins on 5 September in the Main Auditorium.\n\nFinal year "
        "students with no active backlogs may register with the Training and Placement Office.",
        questions=(("Where is the placement drive held?", "the Main Auditorium"),),
    ),
    Sample(
        "notice_sports_day.txt",
        "Notice on Annual Sports Day",
        "notice",
        "Annual Sports Day is on 12 January at the college ground. Classes are suspended that "
        "day.\n\nRegister with the Physical Education Department by 5 January.",
        questions=(("When is Annual Sports Day?", "12 January"),),
    ),
    Sample(
        "notice_bus_routes.txt",
        "Notice on College Bus Routes",
        "notice",
        "Route 4 now stops at Central Station at 7:40 am.\n\nThe bus pass costs 9000 rupees a "
        "year and is issued by the Transport Office.",
        questions=(("How much is the bus pass?", "9000 rupees a year"),),
    ),
    Sample(
        "syllabus_cse_sem3.txt",
        "CSE Semester 3 Syllabus 2026-27",
        "syllabus",
        "Unit 1: Database Design\n\nER modelling, the relational model and relational "
        "algebra.\n\nUnit 2: Normalization\n\nFunctional dependencies, 2NF, 3NF and "
        "BCNF.\n\nUnit 3: Transactions\n\nACID properties, serializability and two-phase "
        "locking.",
        department="CSE",
        semester=3,
        questions=(("Which unit covers BCNF in the CSE third semester?", "Unit 2, Normalization"),),
    ),
    Sample(
        "syllabus_cse_sem5.txt",
        "CSE Semester 5 Syllabus 2026-27",
        "syllabus",
        "Unit 1: Process Management\n\nScheduling, threads and synchronization.\n\nUnit 2: "
        "Memory Management\n\nPaging, segmentation and virtual memory.\n\nUnit 3: Deadlocks\n\n"
        "Conditions, prevention and the banker's algorithm.",
        department="CSE",
        semester=5,
        questions=(("Which unit covers the banker's algorithm?", "Unit 3, Deadlocks"),),
    ),
    Sample(
        "syllabus_ece_sem3.txt",
        "ECE Semester 3 Syllabus 2026-27",
        "syllabus",
        "Unit 1: Network Theory\n\nKirchhoff's laws, Thevenin's theorem and Norton's "
        "theorem.\n\nUnit 2: Signals and Systems\n\nFourier series and the Laplace transform.",
        department="ECE",
        semester=3,
        questions=(
            ("Which ECE unit covers the Laplace transform?", "Unit 2, Signals and Systems"),
        ),
    ),
    Sample(
        "syllabus_mech_sem3.txt",
        "MECH Semester 3 Syllabus 2026-27",
        "syllabus",
        "Unit 1: Thermodynamics\n\nThe first and second laws, entropy and the Carnot "
        "cycle.\n\nUnit 2: Fluid Mechanics\n\nBernoulli's equation and laminar flow.",
        department="MECH",
        semester=3,
        questions=(("Which unit covers the Carnot cycle?", "Unit 1, Thermodynamics"),),
    ),
    Sample(
        "notes_dbms_transactions.txt",
        "Lecture Notes on Transactions",
        "notes",
        "A transaction is a unit of work that either completes fully or has no effect.\n\n"
        "Two-phase locking has a growing phase, which acquires locks, and a shrinking phase, "
        "which releases them. It guarantees conflict serializability.",
        department="CSE",
        semester=3,
        questions=(("What does two-phase locking guarantee?", "conflict serializability"),),
    ),
    Sample(
        "notes_os_paging.txt",
        "Lecture Notes on Paging",
        "notes",
        "Paging divides memory into fixed-size frames, which removes external fragmentation.\n\n"
        "The translation lookaside buffer caches recent page table entries, which makes "
        "address translation faster.",
        department="CSE",
        semester=5,
        questions=(("What does the TLB cache?", "recent page table entries"),),
    ),
    Sample(
        "contacts_offices.txt",
        "Office Contact Directory",
        "contact_list",
        "Examination Cell: 9000000001\n\nOffice of Student Affairs: 9000000002\n\nTransport "
        "Office: 9000000003\n\nTraining and Placement Office: 9000000004",
        questions=(("What is the Examination Cell's phone number?", "9000000001"),),
    ),
    Sample(
        "regulations_antiragging.txt",
        "Anti-Ragging Policy",
        "regulation",
        "Ragging in any form is a punishable offence.\n\nComplaints may be made to the "
        "Anti-Ragging Committee, and are heard within 24 hours. Penalties include suspension "
        "and expulsion.",
        questions=(("How quickly are ragging complaints heard?", "within 24 hours"),),
    ),
    Sample(
        "notice_winter_vacation.txt",
        "Notice on Winter Vacation",
        "notice",
        "Winter vacation runs from 24 December to 2 January.\n\nThe hostel mess stays open "
        "during the vacation for students who register by 15 December.",
        questions=(("When does winter vacation begin?", "24 December"),),
    ),
)
