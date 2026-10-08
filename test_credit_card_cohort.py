"""Policy regression checks on small synthetic duplicate components."""
import unittest
import pandas as pd
from build_credit_card_cohort import Components, decisions, assign_split

class DuplicatePolicyTests(unittest.TestCase):
    def test_later_copy_cannot_replace_earlier_record(self):
        # Lower ID on the future observation cannot trump chronological order.
        f=pd.DataFrame({'Date received':['2025-11-01','2024-12-10','2024-11-10'],
                        'Issue':['Fees or interest']*3},index=['1','2','3'])
        g=Components(f.index); g.union('1','2'); g.union('2','3')
        d=decisions(f,g)
        self.assertEqual(d.at['3','duplicate_status'],'kept')
        self.assertEqual(set(d.representative_id),{'3'})
        self.assertEqual(assign_split(f.at['1','Date received']),'test')
        self.assertEqual(assign_split(f.at['2','Date received']),'validation')

    def test_transitive_conflict_quarantines_every_member_including_rare_label(self):
        f=pd.DataFrame({'Date received':['2024-11-10','2024-12-10','2025-11-01'],
                        'Issue':['Fees or interest','Fees or interest','Improper use of your report']},
                       index=['1','2','3'])
        g=Components(f.index); g.union('1','2')
        self.assertEqual(decisions(f,g).at['1','duplicate_status'],'kept')
        g.union('2','3')
        d=decisions(f,g)
        self.assertEqual(set(d.duplicate_status),{'quarantined_conflict'})
        self.assertEqual(d.group_id.nunique(),1)

    def test_same_day_tie_uses_numeric_id_and_singleton_is_untouched(self):
        f=pd.DataFrame({'Date received':['2024-11-10']*3,'Issue':['Fees or interest']*3},
                       index=['10','2','30'])
        g=Components(f.index); g.union('10','2')
        d=decisions(f,g)
        self.assertEqual(d.at['2','duplicate_status'],'kept')
        self.assertEqual(d.at['10','representative_id'],'2')
        self.assertEqual(d.at['30','duplicate_status'],'kept')

    def test_unassessed_date_fails(self):
        with self.assertRaises(ValueError): assign_split('2025-06-01')

if __name__=='__main__': unittest.main()
